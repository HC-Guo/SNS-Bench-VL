## Step 1: Download Data and Perform Model Inference
First, download the required data and use any inference framework (e.g., vllm, with official website linked) to perform model inference on the data.

### Data Download
Download the data from the [specific data source link]. After downloading, place the data in the data/ directory of the project.

### Model Inference

We recommend using vllm, a high-performance inference framework for large language models. You can obtain more information and installation guidance from its official GitHub repository. 
https://docs.vllm.ai/en/latest/

After installing vllm, use the following command to perform model inference on the downloaded data:

Example vllm inference command  
```shell
python -m server --model [model-name] --port [port-number]  
```

Adjust the model name and port number according to your specific needs.

## Step 2: Configure the Config File
Before proceeding, configure the project’s config file to specify the model path and the path to bge-large-en-1.5.
Configuration Steps
Open the config.yaml file in the project root directory.
Locate and modify the following configuration entries:

```shell
# bge-large-en-1.5 model path configuration  
bge_path: "/path/to/bge-large-en-1.5"  # Replace with the actual path of the bge-large-en-1.5 model  
model_dict:
    your_model_name: "/path/to/your/model"  # Replace with your actual model name and model path  
```


Save the modified config file.

## Step 3: Run the Main Program
After completing the above configurations, run the main program to execute the specific task.
Execution Command
In the project root directory, run the following command:

```shell
python main.py  
```

The final result will be saved in result.cs.

Notes
Ensure all required dependencies are installed. You can install them via:

```shell
pip install -r requirements.txt  
```