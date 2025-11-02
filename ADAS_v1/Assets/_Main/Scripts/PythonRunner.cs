using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics;

public class PythonRunner : MonoBehaviour
{
    [SerializeField]
    string pythonScriptPath = "D:\\Works\\gradschoolworks\\ComputerVision\\TermProjectPy\\TermProjectPy.py";

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

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = "python"; // 혹은 python.exe 의 절대 경로
        psi.Arguments = $"\"{path}\"";
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        Process process = new Process();
        process.StartInfo = psi;
        process.OutputDataReceived += (sender, e) => {
            if (!string.IsNullOrEmpty(e.Data))
            {
                //UnityEngine.Debug.Log(e.Data);
            }
        };
        process.ErrorDataReceived += (sender, e) => {
            if (!string.IsNullOrEmpty(e.Data))
            {
                //UnityEngine.Debug.LogError($"[Python Error] {e.Data}");
            }
        };

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
    }
}
