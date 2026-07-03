Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = "C:\Users\hp\Desktop\newprjt"
shell.Run "cmd /k python run_flask.py", 1, False
