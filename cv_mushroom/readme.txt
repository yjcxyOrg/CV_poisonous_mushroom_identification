蘑菇毒性分类模型（Edible/Poisonous Mushroom Classifier）
项目概述
核心任务是蘑菇图像二分类 输出可食用或有毒结果 可食用对应标签 0 有毒对应标签 1 以安全优先为原则 确保有毒蘑菇召回率不低于 95% 降低漏检风险。技术栈包括 TensorFlow 2.5.0、ResNet50 迁移学习、Matplotlib 和 Seaborn 可视化。运行环境适配 NVIDIA A800 GPU 需搭配 CUDA 11.2 和 cuDNN 8.1 支持 JupyterLab 交互式运行 也可本地部署。
使用方式
1.访问 JupyterLab 共享环境链接 https://a776347-95d2-1839e6cf.nma1.seetacloud.com:8448/jupyter/doc/tree/mushroom进入 mushroom 目录后 可直接使用预置资源 包括数据集（edible 文件夹存放可食用蘑菇图像 poisonous 文件夹存放有毒蘑菇图像）、训练脚本 training3.py、预训练权重、已训练模型 mushroom_model_final_optimized.keras 以及评估图表。
2.本地部署运行
克隆仓库后 创建虚拟环境并安装依赖 执行 pip install -r requirements.txt GPU 环境需先配置 CUDA 11.2 和 cuDNN 8.1。准备数据集 保持 edible 和 poisonous 的目录结构 把两个文件夹放在项目根目录。下载 ResNet50 预训练权重 链接为https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 下载后放入项目根目录。运行训练脚本 执行 python training3.py 训练完成后会生成最终模型和评估图表。预测与部署可通过已训练模型 mushroom_model_final_optimized.h5 进行单图预测 也可启动 Flask 后端 app.py 配合前端 index.html 实现交互式预测。
核心文件说明
edible 和 poisonous 是数据集文件夹 按类别存放蘑菇图像 保持目录结构以适配数据加载逻辑。training3.py 是主训练脚本 包含数据预处理、ResNet50 模型构建、两阶段训练（先冻结基础层再微调顶层）、性能评估和可视化功能。mushroom_model_final_optimized.h5 是最终部署模型 采用 HDF5 格式 支持跨环境加载。resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 是 ResNet50 预训练权重 用于迁移学习的模型初始化。requirements.txt 是依赖清单 记录所有依赖包及对应版本 确保不同环境下的一致性。confusion_matrix.png、recall_history.png 等是评估可视化图表 分别展示混淆矩阵、召回率曲线、阈值分析、精确率 - 召回率曲线等模型性能相关内容。app.py 是 Flask 后端接口脚本 负责接收前端图片请求 调用模型进行预测并返回结果。index.html 是前端交互页面 支持拖拽或点击上传图片 实时展示分类结果。
成果说明

模型在测试集上表现如下 有毒蘑菇召回率不低于 95%，满足安全优先的核心目标。训练完成后会输出两类产物 一是可直接部署的最终模型 二是 4 类评估图表 包括混淆矩阵、召回率曲线、阈值分析图、精确率 - 召回率曲线 这些产物支持后续二次开发或直接部署使用。

使用模型文件（h5格式）对随机从kaggle（https://www.kaggle.com/datasets/lizhecheng/mushroom-classification）获取的数据集进行最后验证，达成了0.74准确率，0.9996召回率的显著成果。
