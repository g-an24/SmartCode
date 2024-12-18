
# SmartCode: Adaptive Code Completion and Bug Fixing with LLaMA 3.1

**SmartCode** is a project to fine-tune **LLaMA 3.1** for adaptive code completion and bug fixing. Inspired by Code Llama's infilling technique, this project aims to automate tedious programming tasks, optimize developer productivity, and improve error detection using real-world code datasets.

---

## 🚀 **Project Goals**

- **Code Completion**: Predict missing or subsequent lines of code given a partially written snippet.
- **Bug Fixing**: Automatically detect and correct errors in Python code.

By fine-tuning LLaMA 3.1 on high-quality datasets, SmartCode delivers state-of-the-art performance for code suggestions and real-time debugging.

---

## 📊 **Dataset**

- The **CodeXGLUE Dataset** is used as the primary data source:
  - **Code Completion Dataset**: Code snippets with partially completed programs.
  - **Bug Fixing Dataset**: Faulty code snippets paired with corrected versions.

- Additional real-world examples are collected using the **GitHub API** for diversity and robustness.

---

## 🔧 **How to Use**

### 1. Prepare Your Dataset

1. Create a Hugging Face account and upload a dataset with a **`content`** column containing code snippets.
   Example datasets can include CodeXGLUE or your own scraped data.

2. Ensure dependencies are installed:
   ```bash
   pip install transformers datasets peft
   ```

### 2. Train the Model on Google Colab

Run the following notebook to train SmartCode with **LLaMA 3.1** using an **A100 GPU** on Google Colab Pro:

Use the **smartcode_fine_tuning.ipynb** and upload it to your Colab Pro environment.

The notebook will:
- Load the dataset.
- Preprocess it for fill-in-the-middle (FIM) tasks.
- Fine-tune LLaMA 3.1 using the Hugging Face `transformers` library.

### 3. Save and Merge the Model

Save the fine-tuned model adapter to your Hugging Face account. Merge it with the base model to deploy a standalone version for inference:

Use the **smartcode_inference.ipynb** and upload it to your Colab Pro environment.

---

## 📈 **Performance Metrics**

SmartCode aims to achieve:
- **85%+ token-level accuracy** for code completion.
- **CodeBLEU score** above **70** for bug fixing tasks.

---

## 💻 **Deploy the Model**

You can deploy the fine-tuned model as an API using **Hugging Face Inference Endpoints** for real-time code completion and debugging.

Example Python Usage:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your_hf_model_name")
model = AutoModelForCausalLM.from_pretrained("your_hf_model_name")

inputs = tokenizer("def factorial(n):", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

---

## 📝 **Future Work**

- Extend support to additional programming languages (Java, C++).
- Explore integration with popular editors like **VSCode** and **PyCharm**.
- Fine-tune on larger datasets to improve bug-fixing accuracy.

---

## 🤝 **Contributors**

- **Ananya Kumar Gangver** (akk8368)
- **Karmanya Mendiratta** (km6296)

---

## 📄 **References**

- **LLaMA 3.1** by Meta: https://github.com/meta-llama
- **CodeXGLUE Benchmark**: https://github.com/microsoft/CodeXGLUE

---



---

## 🎯 **Why SmartCode?**

**SmartCode** is your AI-powered assistant for writing cleaner, bug-free code. By combining fine-tuned LLaMA 3.1 with task-specific datasets, SmartCode simplifies debugging and code generation—saving developers time and effort.

