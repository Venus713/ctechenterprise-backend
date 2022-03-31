# How to built `.exe` file

```bash
 pyinstaller app.py --onefile --name app --copy-metadata sklearn --copy-metadata pandas --hidden-import="sklearn.utils._typedefs" --hidden-import="sklearn.neighbors._partition_nodes"
```
