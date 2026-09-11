Option Explicit

' Double-click this file to launch the supported local browser agent without a terminal window.
Dim shell, filesystem, root
Set shell = CreateObject("WScript.Shell")
Set filesystem = CreateObject("Scripting.FileSystemObject")
root = filesystem.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
shell.Run "cmd /c uv run --extra web agent-web", 0, False
