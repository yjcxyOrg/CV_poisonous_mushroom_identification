import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# ===================== 1. 固定随机种子（确保可复现） =====================
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# ===================== 2. 核心参数配置 =====================
BASE_DIR = "/root/mushroom"  # 数据集根路径
RESIZE_SIZE = (256, 256)
CROP_SIZE = (224, 224)  # 与ResNet50预训练输入尺寸一致
BATCH_SIZE = 64
MAX_EPOCHS_STAGE1 = 50  # 阶段1训练轮次
MAX_EPOCHS_STAGE2 = 30  # 阶段2微调轮次
EARLY_STOP_PATIENCE = 8  # 早停耐心
LR_INIT_STAGE1 = 0.0003
LR_INIT_STAGE2 = LR_INIT_STAGE1 / 5  # 微调阶段学习率降低
MOMENTUM = 0.9
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
VAL_SPLIT_TRAIN = 0.2

# 本地权重文件路径
WEIGHTS_PATH = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"


# ===================== 3. 预处理函数（与预训练逻辑一致） =====================
def preprocess_train(img):
    """训练集预处理：随机裁剪+归一化，增强泛化性"""
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.image.resize(img, RESIZE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.random_crop(img, size=[CROP_SIZE[0], CROP_SIZE[1], 3])
    img = img / 255.0
    img = (img - RGB_MEAN) / RGB_STD
    return img


def preprocess_test_val(img):
    """验证/测试集预处理：固定裁剪+归一化，确保评估稳定"""
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.image.resize(img, RESIZE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.resize_with_crop_or_pad(img, CROP_SIZE[0], CROP_SIZE[1])
    img = img / 255.0
    img = (img - RGB_MEAN) / RGB_STD
    return img


# ===================== 4. 数据生成器（弱化增强，保护正类特征） =====================
def create_generators():
    """创建训练/验证/测试集生成器，调整增强强度以保护蘑菇形态特征"""
    train_val_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=5,  # 减少形态失真
        brightness_range=[0.9, 1.1],  # 保护颜色特征
        validation_split=VAL_SPLIT_TRAIN,
        preprocessing_function=preprocess_train
    )

    train_gen = train_val_datagen.flow_from_directory(
        directory=BASE_DIR,
        target_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['edible', 'poisonous'],  # 0=可食用，1=有毒（正类）
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = train_val_datagen.flow_from_directory(
        directory=BASE_DIR,
        target_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['edible', 'poisonous'],
        subset='validation',
        shuffle=False,
        seed=42
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_test_val)
    test_gen = test_datagen.flow_from_directory(
        directory=BASE_DIR,
        target_size=CROP_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['edible', 'poisonous'],
        shuffle=False,
        seed=42
    )

    print("=" * 60)
    print("数据集生成器信息：")
    print(f"1. 类别映射：{train_gen.class_indices}")
    print(f"2. 数据划分：训练集={train_gen.samples}，验证集={val_gen.samples}，测试集={test_gen.samples}")
    print("=" * 60)
    return train_gen, val_gen, test_gen


train_gen, val_gen, test_gen = create_generators()


# ===================== 5. 构建ResNet50模型 =====================
def build_resnet50():
    """构建两阶段训练模型：先训分类头，再微调顶层"""
    # 检查本地权重文件
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"请下载权重文件至{os.path.abspath(WEIGHTS_PATH)}\n"
            "链接：https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )
    
    # 加载预训练模型（不含分类头）
    base_model = ResNet50(
        weights=WEIGHTS_PATH,
        include_top=False,
        input_shape=(CROP_SIZE[0], CROP_SIZE[1], 3)
    )

    # 阶段1：冻结所有基础层，仅训练分类头
    for layer in base_model.layers:
        layer.trainable = False

    # 分类头设计
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # 二分类输出概率
    model = Model(inputs=base_model.input, outputs=predictions)

    # 阶段1编译
    optimizer = SGD(learning_rate=LR_INIT_STAGE1, momentum=MOMENTUM, clipvalue=1.0)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 Precision(name='precision'), 
                 Recall(name='recall'), 
                 AUC(name='auc')]
    )
    return model, base_model  # 返回base_model用于阶段2解冻


model, base_model = build_resnet50()
print("\nResNet50 模型结构（阶段1：冻结基础层）：")
model.summary()


# ===================== 6. 训练回调函数（阶段1） =====================
callbacks_stage1 = [
    ModelCheckpoint(
        'best_mushroom_model_stage1.keras',
        monitor='val_recall',  # 优先保障有毒蘑菇召回率
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOP_PATIENCE,
        mode='min',
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# ===================== 7. 阶段1：训练分类头（冻结基础层） =====================
print("\n" + "=" * 50)
print("阶段1：训练分类头（冻结ResNet50基础层）")
print("=" * 50)
train_steps = max(1, train_gen.samples // BATCH_SIZE)
val_steps = max(1, val_gen.samples // BATCH_SIZE)

# 计算类别权重：优先提升有毒蘑菇（正类）的学习权重
edible_count = sum(train_gen.classes == 0)
poisonous_count = sum(train_gen.classes == 1)
class_weight_stage1 = {0: poisonous_count / edible_count, 1: 1.0}  # 正类权重更高
print(f"阶段1类别权重：{class_weight_stage1}")

history1 = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=MAX_EPOCHS_STAGE1,
    class_weight=class_weight_stage1,
    callbacks=callbacks_stage1,
    verbose=1
)


# ===================== 8. 阶段2：微调ResNet50最后15层 =====================
print("\n" + "=" * 50)
print("阶段2：微调ResNet50最后15层")
print("=" * 50)
# 解冻最后15层
for layer in base_model.layers[-15:]:
    layer.trainable = True

# 重新编译模型
model.compile(
    optimizer=SGD(learning_rate=LR_INIT_STAGE2, momentum=MOMENTUM, clipvalue=1.0),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             Precision(name='precision'), 
             Recall(name='recall'), 
             AUC(name='auc')]
)

callbacks_stage2 = [
    ModelCheckpoint(
        'best_mushroom_model_stage2.keras',
        monitor='val_recall',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOP_PATIENCE,
        mode='min',
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history2 = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=len(history1.epoch) + MAX_EPOCHS_STAGE2,
    initial_epoch=history1.epoch[-1],
    class_weight=class_weight_stage1,
    callbacks=callbacks_stage2,
    verbose=1
)


# ===================== 9. 阈值评估函数 =====================
def evaluate_with_threshold(model, test_gen, thresholds=np.arange(0.3, 0.6, 0.02)):
    """多阈值搜索，优先保障有毒蘑菇召回率≥95%"""
    y_true = test_gen.classes[:test_gen.samples]  # 真实标签
    y_pred_prob = model.predict(test_gen, verbose=1).flatten()[:len(y_true)]  # 预测概率

    best_f1 = 0.0
    best_thres = 0.5
    target_recall = 0.95  # 业务目标：有毒蘑菇召回率≥95%

    print("\n" + "=" * 70)
    print("Test Set Evaluation with Thresholds (Positive=Poisonous Mushroom)")
    print("=" * 70)
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)

    # 存储所有阈值的评估结果用于可视化
    eval_results = {
        'thresholds': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    for thres in thresholds:
        y_pred = (y_pred_prob > thres).astype(int)
        acc = round(np.mean(y_true == y_pred), 4)
        prec = round(precision_score(y_true, y_pred, zero_division=0), 4)
        rec = round(recall_score(y_true, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)

        # 保存结果
        eval_results['thresholds'].append(thres)
        eval_results['accuracy'].append(acc)
        eval_results['precision'].append(prec)
        eval_results['recall'].append(rec)
        eval_results['f1'].append(f1)

        print(f"{thres:<10.2f} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

        # 筛选逻辑：召回率≥95%且F1最高
        if rec >= target_recall and f1 > best_f1:
            best_f1 = f1
            best_thres = thres

    print("=" * 70)
    print(f"Best Threshold: {best_thres} (Recall≥{target_recall}, F1-Score={best_f1})")
    print("=" * 70)
    return best_thres, eval_results


# ===================== 10. 最终评估 =====================
def final_evaluate(model, test_gen, best_thres, n_repeats=5):
    """重复评估n次，输出平均指标与标准差"""
    y_true = test_gen.classes[:test_gen.samples]
    metrics = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    print("\n" + "=" * 50)
    print(f"Final Evaluation with Threshold {best_thres} (Repeated {n_repeats} Times)")
    print("=" * 50)

    for i in range(n_repeats):
        print(f"\nEvaluation {i+1}/{n_repeats}:")
        y_pred_prob = model.predict(test_gen, verbose=1).flatten()[:len(y_true)]
        y_pred = (y_pred_prob > best_thres).astype(int)

        # 计算指标
        loss = round(model.evaluate(test_gen, verbose=0)[0], 4)
        acc = round(np.mean(y_true == y_pred), 4)
        prec = round(precision_score(y_true, y_pred, zero_division=0), 4)
        rec = round(recall_score(y_true, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_true, y_pred, zero_division=0), 4)
        auc = round(tf.keras.metrics.AUC()(y_true, y_pred_prob).numpy(), 4)

        metrics['loss'].append(loss)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)
        metrics['auc'].append(auc)

    # 输出平均结果
    print("\n" + "=" * 70)
    print(f"Average Metrics with Threshold {best_thres}")
    print("=" * 70)
    for k, v in metrics.items():
        mean_val = round(np.mean(v), 4)
        std_val = round(np.std(v), 4)
        print(f"{k.capitalize()}: {mean_val} ± {std_val}")
    return metrics, y_true, y_pred_prob


# ===================== 11. 可视化函数扩展 - 重点增强召回率相关图表 =====================
def plot_recall_history(h1, h2):
    """单独绘制召回率训练曲线，突出展示val-recall"""
    recall = h1.history['recall'] + h2.history['recall']
    val_recall = h1.history['val_recall'] + h2.history['val_recall']
    epochs = range(len(recall))
    fine_tune_start = len(h1.epoch) - 1  # 微调开始的epoch

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, recall, 'b-', label='Training Recall', linewidth=2)
    plt.plot(epochs, val_recall, 'r-', label='Validation Recall', linewidth=2)
    plt.axvline(fine_tune_start, color='gray', linestyle='--', label='Start Fine-tuning')
    
    # 标记最佳验证召回率点
    best_val_recall = max(val_recall)
    best_epoch = val_recall.index(best_val_recall)
    plt.scatter(best_epoch, best_val_recall, color='green', s=100, zorder=5)
    plt.annotate(f'Best: {best_val_recall:.4f}', 
                 xy=(best_epoch, best_val_recall),
                 xytext=(10, 10),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))

    plt.title('Training and Validation Recall for Poisonous Mushrooms', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.ylim(0.8, 1.0)  # 聚焦在高召回率区域
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('recall_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("召回率训练曲线已保存为: recall_history.png")


def plot_threshold_analysis(eval_results, best_thres, target_recall):
    """绘制不同阈值下的召回率、精确率和F1分数变化"""
    plt.figure(figsize=(12, 6))
    
    # 绘制召回率曲线
    plt.plot(eval_results['thresholds'], eval_results['recall'], 'g-', 
             label='Recall', linewidth=2)
    # 绘制精确率曲线
    plt.plot(eval_results['thresholds'], eval_results['precision'], 'b-', 
             label='Precision', linewidth=2)
    # 绘制F1曲线
    plt.plot(eval_results['thresholds'], eval_results['f1'], 'purple', 
             label='F1 Score', linewidth=2)
    
    # 标记目标召回率线和最佳阈值
    plt.axhline(y=target_recall, color='r', linestyle='--', 
                label=f'Target Recall ({target_recall})')
    plt.axvline(x=best_thres, color='orange', linestyle='--', 
                label=f'Best Threshold ({best_thres:.2f})')
    
    plt.title('Metrics vs. Threshold Values', fontsize=14)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("阈值分析图表已保存为: threshold_analysis.png")


def plot_precision_recall_curve(y_true, y_pred_prob):
    """绘制精确率-召回率曲线"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.grid(alpha=0.3)
    
    # 标记95%召回率位置
    target_recall = 0.95
    idx = np.argmin(np.abs(recall - target_recall))
    plt.scatter(recall[idx], precision[idx], color='red', s=100, zorder=5)
    plt.annotate(f'95% Recall: {precision[idx]:.4f}', 
                 xy=(recall[idx], precision[idx]),
                 xytext=(-100, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("精确率-召回率曲线已保存为: precision_recall_curve.png")


def plot_confusion_matrix_analysis(y_true, y_pred_prob, best_thres):
    """绘制混淆矩阵"""
    y_pred = (y_pred_prob > best_thres).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Edible', 'Poisonous'],
                yticklabels=['Edible', 'Poisonous'])
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("混淆矩阵已保存为: confusion_matrix.png")


def plot_training_curves(h1, h2):
    """合并阶段1和阶段2的训练曲线"""
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    precision = h1.history['precision'] + h2.history['precision']
    val_precision = h1.history['val_precision'] + h2.history['val_precision']
    epochs = range(len(acc))
    fine_tune_start = len(h1.epoch) - 1

    # 创建2x2子图
    plt.figure(figsize=(15, 12))
    
    # 子图1：Accuracy
    plt.subplot(221)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.axvline(fine_tune_start, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.title('Training & Validation Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    # 子图2：Loss
    plt.subplot(222)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.axvline(fine_tune_start, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.title('Training & Validation Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Binary Crossentropy Loss', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    # 子图3：Precision
    plt.subplot(223)
    plt.plot(epochs, precision, 'b-', label='Training Precision')
    plt.plot(epochs, val_precision, 'r-', label='Validation Precision')
    plt.axvline(fine_tune_start, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.title('Training & Validation Precision', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Precision', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("训练曲线已保存为: training_history.png")


# ===================== 12. 执行评估与可视化 =====================
best_threshold, eval_results = evaluate_with_threshold(model, test_gen)
final_metrics, y_true, y_pred_prob = final_evaluate(model, test_gen, best_threshold)

# 生成所有可视化图表
plot_training_curves(history1, history2)
plot_recall_history(history1, history2)  # 重点展示val-recall
plot_threshold_analysis(eval_results, best_threshold, target_recall=0.95)
plot_precision_recall_curve(y_true, y_pred_prob)
plot_confusion_matrix_analysis(y_true, y_pred_prob, best_threshold)

# 保存最终模型
model.save('mushroom_model_final_optimized.keras')
print(f"\n最终模型已保存为: mushroom_model_final_optimized.keras (最佳阈值: {best_threshold})")
print("\n" + "=" * 80)
print("蘑菇分类模型（可食用/有毒）训练完成!")
print("=" * 80)
