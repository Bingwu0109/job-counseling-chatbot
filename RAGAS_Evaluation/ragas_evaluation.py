import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Configure model paths (modify according to your actual paths)
MODEL_PATH = "path/to/your/qwen1.5-7B-chat"  # Replace with your Qwen model path
EMBEDDING_MODEL_PATH = "path/to/your/BAAI/bge-large-en-v1.5"  # Replace with your embedding model path


def setup_local_models():
    """Setup local models"""
    print("Loading local models...")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Qwen model as LLM
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        device=0 if device == "cuda" else -1
    )

    # Wrap as LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("Models loaded successfully!")
    return llm, embeddings


def load_and_prepare_data(csv_path):
    """加载和准备数据"""
    print("正在加载CSV数据...")

    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 检查必需的列是否存在
    required_columns = ['questions', 'answers', 'ground_truth']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"CSV文件缺少以下列: {missing_columns}")

    # 准备RAGAS格式的数据
    # 注意：RAGAS需要contexts字段用于Context Relevancy评估
    # 这里将ground_truth作为contexts使用
    ragas_data = {
        'question': df['questions'].tolist(),
        'answer': df['answers'].tolist(),
        'contexts': [[str(context)] for context in df['ground_truth'].tolist()],  # 转换为列表格式
        'ground_truth': df['ground_truth'].tolist()  # 保留原始ground_truth
    }

    # 创建Dataset对象
    dataset = Dataset.from_dict(ragas_data)

    print(f"数据加载完成，共{len(dataset)}条记录")
    return dataset


def run_ragas_evaluation(dataset, llm, embeddings):
    """Run RAGAS evaluation"""
    print("Starting RAGAS evaluation...")

    # Configure evaluation metrics
    metrics = [
        faithfulness,  # Faithfulness
        answer_relevancy,  # Answer Relevancy
        context_relevancy  # Context Relevancy
    ]

    # Set up RAGAS models
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Wrap models
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Set models for each metric
    for metric in metrics:
        metric.llm = ragas_llm
        metric.embeddings = ragas_embeddings

    try:
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )

        print("Evaluation completed!")
        return result

    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        return None


def save_results(result, output_path="ragas_evaluation_results.csv"):
    """Save evaluation results"""
    if result is None:
        print("No results to save")
        return

    # Convert results to DataFrame
    results_df = result.to_pandas()

    # Save to CSV
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to: {output_path}")

    # Print summary statistics
    print("\n=== Evaluation Results Summary ===")
    for column in results_df.columns:
        if column not in ['question', 'answer', 'contexts', 'ground_truth']:
            mean_score = results_df[column].mean()
            print(f"{column}: {mean_score:.4f}")

    return results_df


def main():
    """Main function"""
    # Configure file paths
    CSV_PATH = "your_qa_data.csv"  # Replace with your CSV file path
    OUTPUT_PATH = "ragas_evaluation_results.csv"  # Output file path

    try:
        # 1. Setup models
        llm, embeddings = setup_local_models()

        # 2. Load data
        dataset = load_and_prepare_data(CSV_PATH)

        # 3. Run evaluation
        result = run_ragas_evaluation(dataset, llm, embeddings)

        # 4. Save results
        results_df = save_results(result, OUTPUT_PATH)

        print("\nEvaluation process completed!")

    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()


# Helper function: Check data format
def preview_data(csv_path, num_rows=3):
    """Preview data format"""
    df = pd.read_csv(csv_path)
    print("=== Data Preview ===")
    print(f"Data shape: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    print(f"\nFirst {num_rows} rows:")
    print(df.head(num_rows))
    return df


if __name__ == "__main__":
    # Uncomment the line below if you need to preview data first
    # preview_data("your_qa_data.csv")

    # Run main program
    main()