import pandas as pd
import os

def main():
    # Take all of each complexity and put it together
    models = ['deepseek-coder-6.7b-instruct', 'Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-Coder-7B-Instruct']
    dt = {'ptl': [2, 3, 4], 'sll': [1, 2, 3], 'ml': ['Easy', 'Medium', 'Hard']}
    for c in dt:
        for b in dt[c]:
            df = pd.DataFrame()
            for model in models:
                data = pd.read_csv(f"results/lc/{model}-infilled-{c}-{b}.csv")
                data.insert(0, "Model", model)
                df = pd.concat([df, data], ignore_index=True)

            os.makedirs("results/lc", exist_ok=True)
            df.to_csv(f"results/lc/PrePostSplit-infilled-{c}-{b}.csv", index=False)


if __name__ == "__main__":
    main()