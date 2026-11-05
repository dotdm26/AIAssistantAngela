#define MyAppName "Angela"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Angela"
#define MyAppExeName "Angela.exe"

[Setup]
AppId={{A7E2E1DE-DC65-4F5F-92AC-9C79E8FA0A6E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Angela
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\..\dist\installer
OutputBaseFilename=Angela-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\..\dist\Angela\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Angela"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Angela"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Angela"; Flags: nowait postinstall skipifsilent

; User data intentionally lives outside {app} and is not removed by the
; uninstaller. This preserves credentials, conversations, and configuration.
