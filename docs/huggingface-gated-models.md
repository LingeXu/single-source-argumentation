# 使用受限 Hugging Face 模型提交 🔒

目标是确保你的模型保持安全且私密，同时又可以被评估服务器访问。你可以通过将模型包装为公开可见但安全受限的 Hugging Face 仓库来实现这一点。

要安全地使用此类受限公开 Hugging Face 模型进行提交，你必须授予 `aicrowd` 账号访问你的公开但受限仓库的权限。**所有仓库名称必须包含 "aicrowd"**，以确保验证成功。

- ✅ **有效示例**：`team-aicrowd-my-model`
- ❌ **无效示例**：`team-my-model`

---

## 推荐的团队配置

- **单人团队**：在你的个人 Hugging Face 账号下创建公开受限模型
- **多人团队**：创建一个 Hugging Face 组织，在该组织内管理公开受限模型，以便更好地进行团队协作和协调

**注意**：公开受限模型确保你的模型是安全的。只有明确被邀请的账号（如 `aicrowd`）才有访问权限，其他参赛者无法查看或访问你的提交。

---

## 分步指南：创建公开受限 Hugging Face 模型

1. 登录你的 [Hugging Face](https://huggingface.co/) 账号
2. 点击 **New Model**
3. 输入模型名称（必须包含 "aicrowd"），将可见性设为 **Public**，点击 **Create Model**
4. 进入模型页面，点击 **Settings** 标签
5. 在 **Access Control** 下，启用 **"Enable Access Requests"** 来限制你的模型
6. 点击 **Save** 应用更改

---

## 授予 `aicrowd` 访问权限

按以下步骤立即授予 `aicrowd` 账号访问权限：

1. 进入 Hugging Face 上模型的设置页面
2. 在 **Settings** 下，启用 **Access Requests**（如果尚未启用）
3. 点击 **Add Access**，搜索 **aicrowd**，选择该账号，点击 **Grant Access**

> **注意**：此过程会**立即**授予 `aicrowd` 账号访问权限。

---

## 在 `aicrowd.json` 中指定你的模型

在 `aicrowd.json` 文件中明确指定你的模型：

```json
"hf_models": [
    {
        "repo_id": "your-hf-username/team-aicrowd-my-model",
        "revision": "main"
    }
]
```

---

## 重要提醒

- 未明确授予 `aicrowd` 账号访问权限将导致提交失败
- 确保你的仓库名称始终包含关键词 "aicrowd"
