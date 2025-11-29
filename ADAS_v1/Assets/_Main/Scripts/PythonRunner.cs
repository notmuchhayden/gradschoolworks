using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;
using System.IO;

public class PythonRunner : MonoBehaviour
{
    [SerializeField]
    // 지정할 때는 Assets 폴더 기준의 상대 경로(또는 Assets 없이 시작하는 경로)를 사용하세요.
    string pythonScriptPath = "_Main/YOLOv8/TermProjectPy.py";

    // Start is called before the first frame update
    void Start()
    {
        RunPythonScript(pythonScriptPath);
    }

    void RunPythonScript(string path) 
    { 
        // 경로가 null, 빈 문자열 또는 공백 문자만으로 이루어져 있으면 실행하지 않음
        if (string.IsNullOrWhiteSpace(path))
        {
            UnityEngine.Debug.LogWarning("RunPythonScript: path가 비어있습니다. 실행을 건너뜁니다.");
            return;
        }

        // 절대경로가 아니면 Unity 프로젝트의 Assets 폴더를 기준으로 절대경로로 변환
        string resolvedPath = path;
        if (!Path.IsPathRooted(path))
        {
            // 통일된 구분자를 사용하고, 만약 경로가 "Assets/..." 로 시작하면 중복을 피함
            string normalized = path.Replace('\\', '/').TrimStart('/');
            if (normalized.StartsWith("Assets/"))
            {
                normalized = normalized.Substring("Assets/".Length);
            }
            resolvedPath = Path.Combine(Application.dataPath, normalized);
        }

        if (!File.Exists(resolvedPath))
        {
            UnityEngine.Debug.LogError($"RunPythonScript: 파일을 찾을 수 없음: {resolvedPath}");
            return;
        }

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = "python"; // 혹은 python.exe 의 절대 경로
        psi.Arguments = $"\"{resolvedPath}\"";
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        Process process = new Process();
        process.StartInfo = psi;
        process.OutputDataReceived += (sender, e) => {
            if (!string.IsNullOrEmpty(e.Data))
            {
                UnityEngine.Debug.Log(e.Data);
            }
        };
        process.ErrorDataReceived += (sender, e) => {
            if (!string.IsNullOrEmpty(e.Data))
            {
                UnityEngine.Debug.LogError($"[Python Error] {e.Data}");
            }
        };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
    }
}
