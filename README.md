# 矿井安全检查系统（后端大模型工程师编程题）

基于 YOLOv8 明火检测的矿井安全检查系统，含 FastAPI 后端 API。

## 项目结构

```
├── module1_yolo/               # YOLOv8 训练与推理模块
│   ├── config.yaml             # 训练配置
│   ├── config.py               # 配置加载器
│   ├── data_converter.py       # Kaggle数据集→YOLO格式转换
│   ├── train.py                # 迁移学习训练脚本
│   └── predict.py              # 推理脚本
├── module2_backend/            # FastAPI 后端
│   ├── app.py                  # 应用入口
│   ├── core/
│   │   ├── config.py           # 应用配置
│   │   └── database.py         # DB会话管理
│   ├── models/
│   │   └── database.py         # SQLModel 数据表定义
│   ├── schemas/
│   │   └── inspection.py       # Pydantic 请求/响应模型
│   ├── services/
│   │   └── inspection_service.py  # CRUD 业务逻辑层
│   ├── routers/
│   │   └── inspections.py      # API 路由
│   └── ml/
│       └── detector.py         # YOLO 模型推理封装
├── tests/
│   └── test_api.py             # API 测试
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 API 服务

```bash
uvicorn module2_backend.app:app --reload --port 8004
```

访问 http://localhost:8004/docs 查看 Swagger 文档

### 3. YOLOv8 模块

**数据集准备：**
```bash
# 下载 Kaggle 明火数据集后转换格式
python module1_yolo/data_converter.py --input /path/to/kaggle_fire --output ./data/fire_dataset
```

**训练（迁移学习）：**
```bash
python module1_yolo/train.py
```

**推理：**
```bash
python module1_yolo/predict.py --source path/to/fire_image.jpg
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/inspections/` | 提交照片进行安全检查 |
| GET | `/api/v1/inspections/` | 分页查询检查记录 |
| GET | `/api/v1/inspections/{id}` | 查询单条记录 |
| DELETE | `/api/v1/inspections/{id}` | 删除记录 |
| GET | `/health` | 健康检查 |

### POST 参数

| 字段 | 类型 | 说明 |
|------|------|------|
| inspection_date | string | 检查日期 (YYYY-MM-DD) |
| team_id | int | 施工队编号 |
| area_id | int | 采区编号 |
| shift | string | 班组 |
| photo | file | 图片文件 |
| remark | string | 备注(可选) |

### 查询参数

| 参数 | 类型 | 说明 |
|------|------|------|
| page | int | 页码 (默认1) |
| page_size | int | 每页条数 (默认20) |
| team_id | int | 施工队筛选(可选) |
| area_id | int | 采区筛选(可选) |

## 运行测试

```bash
pytest tests/
```
