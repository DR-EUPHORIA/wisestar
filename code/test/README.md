## 智多星Test代码详解

### Work_Flow

- 该文件实现所有工作流的实现，可以自动读取Node结尾的节点，便捷实现工作流的实现
- 目前主要通过分支来进行工作流的判断，因此需要在if分支中添加agent框架名称
- 已经实现RePI、ReflectPI、DeRePI的适配，可以在环境变量中直接添加 MODULE_TYPE 进行agent框架的选择

### Test_async

- 该文件主要实现外部并发，实现多任务的运行
- 此处的Prompt全部储存在utils中的prompt_templates
- 在制作数据集中请保证包含以下***tag***
- ID请依次递增，否则无法断点调试

1. **model_answer**
2. **ground_truth**
3. **question or question（纯文本）**


### 附.env模版
* IDEALAB_API_KEY：apikey
* MODEL_NAME：模型名称，选qwen系列 //模型列表见：https://aliyuque.antfin.com/aiplus/aistudio/gy7m97kce97gga33#iA7X
* MAX_RETRY：ReNode调用最大重试数
* CONCURRENCY_LIMIT：异步批处理/外置并发 的并发数
* TIMEOUT：PINode的代码执行最大时间，超过则停止运行，防止堵塞
* FILE_NAME: 数据库名称（不加后缀）
* MODULE_TYPE: 训练框架选择（RePI、ReflectPI等）