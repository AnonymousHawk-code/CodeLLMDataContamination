import ast
import random

class RenameVariables(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.changed = False  # Track if any change happened

    def visit_FunctionDef(self, node):
        self.param_map = {}

        # Rename each parameter and record the mapping
        idx = 0
        for arg in node.args.args:
            old_name = arg.arg
            if old_name == "self":
                continue  # Skip renaming 'self'
            new_name = f"var_{idx}"
            self.param_map[old_name] = new_name
            arg.arg = new_name
            self.changed = True
            idx += 1  # Only increment idx for renamed params

        self.generic_visit(node)  # Apply mapping to function body
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Del)) and node.id in getattr(self, 'param_map', {}):
            return ast.copy_location(
                ast.Name(id=self.param_map[node.id], ctx=node.ctx),
                node
            )
        return node


class RenameCallKeywords(ast.NodeTransformer):
    def __init__(self):
        self.changed = False

    def visit_Call(self, node):
        # Only rename keyword arguments (e.g., s="abc")
        for idx, kw in enumerate(node.keywords):
            kw.arg = f"var_{idx}"
            self.changed = True
        self.generic_visit(node)
        return node

class KeywordToPositional(ast.NodeTransformer):
    def visit_Call(self, node):
        # Move keyword arguments to positional, in the same order
        for kw in node.keywords:
            node.args.append(kw.value)
        node.keywords = []  # Clear keyword args
        return self.generic_visit(node)


class InsertIdentity(ast.NodeTransformer):
    def __init__(self, param_name):
        super().__init__()
        self.param_name = param_name
        self.inserted = False
        self.insert_lineno = None  # will hold insertion line number

    def visit_FunctionDef(self, node):
        if self.inserted:
            return node

        # Find insertion index (after docstring if present)
        insert_idx = 0
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            insert_idx = 1

        # Determine line number after which we insert
        if insert_idx > 0:
            # Insert after docstring statement line
            self.insert_lineno = node.body[insert_idx - 1].lineno
        else:
            # Insert after function def line (approximate)
            self.insert_lineno = node.lineno

        op = ast.Mult()
        val = ast.Constant(value=1)

        identity_stmt = ast.Assign(
            targets=[ast.Name(id=self.param_name, ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id=self.param_name, ctx=ast.Load()),
                op=op,
                right=val
            )
        )

        # Manually set lineno for inserted statement (approximate)
        identity_stmt.lineno = self.insert_lineno + 0.1  # fractional so it's between lines
        identity_stmt.col_offset = 0

        node.body.insert(insert_idx, identity_stmt)
        self.inserted = True
        return node


class ForToWhileTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.changed = False  # Track if any change happened

    def visit_For(self, node):
        # Only handle `for i in range(N)` loops
        if isinstance(node.iter, ast.Call) and getattr(node.iter.func, 'id', None) == 'range':
            var = node.target.id
            range_arg = node.iter.args[0]

            # Create: i = 0
            init = ast.Assign(
                targets=[ast.Name(id=var, ctx=ast.Store())],
                value=ast.Constant(value=0)
            )

            # Create: while i < N:
            test = ast.Compare(
                left=ast.Name(id=var, ctx=ast.Load()),
                ops=[ast.Lt()],
                comparators=[range_arg]
            )

            # Create: i += 1
            incr = ast.AugAssign(
                target=ast.Name(id=var, ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(value=1)
            )

            new_body = node.body + [incr]

            while_node = ast.While(test=test, body=new_body, orelse=[])

            self.changed = True

            # Return init + while
            return [init, while_node]

        # Only handle for-loop iterates over enumerate(...)
        if (isinstance(node.iter, ast.Call) and
                getattr(node.iter.func, 'id', None) == 'enumerate' and
                len(node.iter.args) == 1):

            iterable = node.iter.args[0]

            # Expect target is a tuple: for i, val in enumerate(...)
            if (isinstance(node.target, ast.Tuple) and
                    len(node.target.elts) == 2 and
                    all(isinstance(e, ast.Name) for e in node.target.elts)):
                index_var = node.target.elts[0].id  # e.g. 'i'
                value_var = node.target.elts[1].id  # e.g. 'val'

                # i = 0
                init = ast.Assign(
                    targets=[ast.Name(id=index_var, ctx=ast.Store())],
                    value=ast.Constant(value=0)
                )

                # while i < len(iterable):
                test = ast.Compare(
                    left=ast.Name(id=index_var, ctx=ast.Load()),
                    ops=[ast.Lt()],
                    comparators=[ast.Call(
                        func=ast.Name(id='len', ctx=ast.Load()),
                        args=[iterable],
                        keywords=[]
                    )]
                )

                # val = iterable[i]
                assign_val = ast.Assign(
                    targets=[ast.Name(id=value_var, ctx=ast.Store())],
                    value=ast.Subscript(
                        value=iterable,
                        slice=ast.Index(ast.Name(id=index_var, ctx=ast.Load())),
                        ctx=ast.Load()
                    )
                )

                # i += 1
                incr = ast.AugAssign(
                    target=ast.Name(id=index_var, ctx=ast.Store()),
                    op=ast.Add(),
                    value=ast.Constant(value=1)
                )

                # New body = val assignment + original body + i increment
                new_body = [assign_val] + node.body + [incr]

                while_node = ast.While(test=test, body=new_body, orelse=[])

                self.changed = True

                # Replace For node with [init, while_node]
                return [init, while_node]

        return node


class ForInToRangeTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.changed = False  # Track if any change happened

    def visit_For(self, node):
        # Only transform basic: for x in iterable
        if isinstance(node.iter, ast.expr) and isinstance(node.target, ast.Name):
            iterable = node.iter
            target = node.target.id
            index_var = "i"

            # Create: for _i in range(len(iterable))
            new_for = ast.For(
                target=ast.Name(id=index_var, ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id="len", ctx=ast.Load()),
                            args=[iterable],
                            keywords=[]
                        )
                    ],
                    keywords=[]
                ),
                body=[
                         ast.Assign(
                             targets=[ast.Name(id=target, ctx=ast.Store())],
                             value=ast.Subscript(
                                 value=iterable,
                                 slice=ast.Index(value=ast.Name(id=index_var, ctx=ast.Load())),
                                 ctx=ast.Load()
                             )
                         )
                     ] + node.body,
                orelse=node.orelse
            )

            self.changed = True

            return ast.fix_missing_locations(new_for)

        return node


class IfElseInverter(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.changed = False  # Track if any change happened

    def visit_If(self, node):
        self.generic_visit(node)  # Recursively visit children first

        # Only transform if there is an else block
        if node.orelse:
            # Invert the condition
            new_test = ast.UnaryOp(
                op=ast.Not(),
                operand=node.test
            )

            # Swap body and orelse
            node.test = new_test
            node.body, node.orelse = node.orelse, node.body

            ast.fix_missing_locations(node)

            self.changed = True

        return node