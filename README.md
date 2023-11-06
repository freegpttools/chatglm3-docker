# How to quickly deploy a demo of ChatGLM3 using Docker, Docker Compose, and Poetry in an environment without a GPU.

**1. Create a folder named `build` and create a file named `Dockerfile` inside it with the following content:**

```Dockerfile
FROM python:3.10-bookworm

RUN apt update 
RUN apt install -y ca-certificates curl
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.in-project true
```

With the above `Dockerfile`, you can create an image based on Python 3.10 that includes Poetry. Additionally, by executing `poetry config virtualenvs.in-project true`, you ensure that Poetry creates virtual Python environments and installs dependencies in the project folder.

**2. Create a file named `docker-compose.yml` and add the following content:**

```yaml
version: "3"
services:
  py:
    build:
      context: ./build

    volumes:
      - "./ChatGLM3:/app"
    working_dir: /app
    environment:
      - TRANSFORMERS_CACHE=.cache
    ports:
      - 8501:8501
    command:
      - poetry
      - run
      - streamlit
      - run
      - web_demo2.py
```

By declaring `TRANSFORMERS_CACHE` as `.cache`, you ensure that model files are downloaded to the `.cache` folder in the project directory.

**3. Clone the ChatGLM3 project from [GitHub](https://github.com/THUDM/ChatGLM3):**

```bash
git clone https://github.com/THUDM/ChatGLM3
```

**4. Create a file named `pyproject.toml` in the `ChatGLM3` folder and add the following content:**

```toml
[tool.poetry]
name = "app"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
protobuf = "^4.24.4"
transformers = "4.30.2"
cpm-kernels = "^1.0.11"
gradio = "3.39"
mdtex2html = "^1.2.0"
sentencepiece = "^0.1.99"
accelerate = "^0.24.1"
sse-starlette = "^1.6.5"
streamlit = ">=1.24.0"
fastapi = "0.95.1"
typing-extensions = "4.4.0"
uvicorn = "^0.23.2"
torch = {version = "^2.1.0+cpu", source = "pytorch"}

[[tool.poetry.source]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
priority = "explicit"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

This file is the project's Poetry configuration, declaring all the required dependencies and setting the `index-url` for PyTorch to "https://download.pytorch.org/whl/cpu".

**5. Modify `ChatGLM3/web_demo2.py`**

Replace the contents of `ChatGLM3/web_demo2.py` with the following code:

```python
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer

# 设置页面标题、图标和布局
st.set_page_config(
    page_title="ChatGLM3-6B 演示",
    page_icon=":robot:",
    layout="wide"
)

# 设置为模型ID或本地文件夹路径
model_path = "THUDM/chatglm3-6b"

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()
    # 多显卡支持,使用下面两行代替上面一行,将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm3-6b", num_gpus=2)
    model = model.eval()
    return tokenizer, model

# 加载Chatglm3的model和tokenizer
tokenizer, model = get_model()

# 初始化历史记录和past key values
if "history" not in st.session_state:
    st.session_state.history = []
if "past_key_values" not in st.session_state:
    st.session_state.past_key_values = None

# 设置max_length、top_p和temperature
max_length = st.sidebar.slider("max_length", 0, 32768, 8192, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.6, step=0.01)

# 清理会话历史
buttonClean = st.sidebar.button("清理会话历史", key="clean")
if buttonClean:
    st.session_state.history = []
    st.session_state.past_key_values = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.rerun()

# 渲染聊天历史记录
for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        with st.chat_message(name="user", avatar="user"):
            st.markdown(message["content"])
    else:
        with st.chat_message(name="assistant", avatar="assistant"):
            st.markdown(message["content"])

# 输入框和输出框
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

# 获取用户输入
prompt_text = st.chat_input("请输入您的问题")

# 如果用户输入了内容,则生成回复
if prompt_text:

    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    past_key_values = st.session_state.past_key_values
    for response, history, past_key_values in model.stream_chat(
        tokenizer,
        prompt_text,
        history,
        past_key_values=past_key_values,
        max_length=max_length,
        top_p=top_p,
        temperature=temperature,
        return_past_key_values=True,
    ):
        message_placeholder.markdown(response)

    # 更新历史记录和past key values
    st.session_state.history = history
    st.session_state.past_key_values = past_key_values
```

This file modifies the original `model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()` to `model = AutoModel.from_pretrained(model_path, trust_remote_code=True).float()` to make it run in an environment without a GPU.

Now, all the necessary configurations are in place. You can proceed to install dependencies and start the service.

**6. Install Dependencies**

Install the project's required dependencies with the following command:

```bash
docker-compose run --rm py poetry install
```

**7. Start the Service**

Start the service with the following command:

```bash
docker-compose up -d
```

You may need to wait for some time as the service downloads the model files. Once it's ready, you can access the deployed service's interface in your web browser at `http://localhost:8501/`. If you're deploying remotely, replace `localhost` with the appropriate IP address.

**8. Start Chatting**

Now you can start chatting with ChatGLM3 using the service you've deployed!

For more content, please visit: [https://www.freegpttools.org/chatglm3](https://www.freegpttools.org/chatglm3)