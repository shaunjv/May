Option Explicit

' Double-click this file to launch the supported local browser agent without a terminal window.
Dim shell, filesystem, root, command
Set shell = CreateObject("WScript.Shell")
Set filesystem = CreateObject("Scripting.FileSystemObject")
root = filesystem.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root

' If the local app is already running, simply open it. Otherwise start it hidden;
' agent-web opens the browser after the server is ready.
command = "powershell.exe -NoProfile -WindowStyle Hidden -Command ""$running = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue; if ($running) { Start-Process 'http://127.0.0.1:8765/' } else { Start-Process -FilePath 'uv' -ArgumentList @('run','--extra','web','agent-web') -WorkingDirectory '" & Replace(root, "'", "''") & "' -WindowStyle Hidden }"""
shell.Run command, 0, False
