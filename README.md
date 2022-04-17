#

## How to built `.exe` file

```bash
 pyinstaller app.py --onefile --name app --copy-metadata sklearn --copy-metadata pandas --hidden-import="sklearn.utils._typedefs" --hidden-import="sklearn.neighbors._partition_nodes"
```

## How to run the `.exe` file

```bash
app.exe -f unsw-train-1k.csv -c "0 1 2 3 4 5 8 9 11 23 25 29 35 39"
```
