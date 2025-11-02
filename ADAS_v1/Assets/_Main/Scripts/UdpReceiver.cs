using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class Detection
{
    // Internal representation used by Unity
    public int classId;
    public float confidence;
    // bounding box in xyxy pixel coords: x1,y1,x2,y2
    public float x1;
    public float y1;
    public float x2;
    public float y2;
}

// Matches the Python sender JSON structure: {"class": int, "confidence": float, "bbox": [x1,y1,x2,y2]}
[Serializable]
public class DetectionRaw
{
    // `class` is a C# keyword, use verbatim identifier so JsonUtility can map the JSON key
    public int @class;
    public float confidence;
    public float[] bbox;
}

[Serializable]
public class DetectionRawList
{
    public DetectionRaw[] detections;
}

public class UdpReceiver : MonoBehaviour
{
    [Tooltip("UDP port to listen on")]
    public int listenPort = 5005; // matches Python sender default

    // Latest detections parsed on main thread
    public List<Detection> latestDetections = new List<Detection>();

    // Optional action subscribers can use
    public event Action<List<Detection>> OnDetectionsReceived;

    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool running;

    // Internal queue for thread -> main thread handoff
    private readonly Queue<string> messageQueue = new Queue<string>();
    private readonly object queueLock = new object();

    void Start()
    {
        StartListener();
    }

    void Update()
    {
        // Drain queued messages on main thread and parse
        while (true)
        {
            string msg = null;
            lock (queueLock)
            {
                if (messageQueue.Count > 0)
                {
                    msg = messageQueue.Dequeue();
                }
            }

            if (msg == null) break;

            try
            {
                // The Python sender sends a top-level JSON array like:
                // [{"class":0,"confidence":0.9,"bbox":[x1,y1,x2,y2]}, ...]
                // Unity's JsonUtility cannot parse top-level arrays, so wrap it into an object:
                string wrapped = msg.TrimStart();
                if (wrapped.StartsWith("["))
                {
                    wrapped = "{\"detections\":" + msg + "}";
                }

                var rawList = JsonUtility.FromJson<DetectionRawList>(wrapped);

                latestDetections.Clear();
                if (rawList != null && rawList.detections != null)
                {
                    foreach (var r in rawList.detections)
                    {
                        var d = new Detection();
                        d.classId = r.@class;
                        d.confidence = r.confidence;
                        if (r.bbox != null && r.bbox.Length >= 4)
                        {
                            d.x1 = r.bbox[0];
                            d.y1 = r.bbox[1];
                            d.x2 = r.bbox[2];
                            d.y2 = r.bbox[3];
                        }
                        latestDetections.Add(d);
                    }
                }
                // latestDetections 의 첫번째 내용을 Log 로 출력
                UnityEngine.Debug.Log($"UdpReceiver: Received {latestDetections.Count} detections. First detection: " +
                    (latestDetections.Count > 0 ? 
                    $"classId={latestDetections[0].classId}, confidence={latestDetections[0].confidence}, " +
                    $"bbox=({latestDetections[0].x1}, {latestDetections[0].y1}, {latestDetections[0].x2}, {latestDetections[0].y2})" 
                    : "N/A"));


                OnDetectionsReceived?.Invoke(latestDetections);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("UdpReceiver: Failed to parse message: " + ex.Message);
            }
        }
    }

    void OnDestroy()
    {
        StopListener();
    }

    void OnApplicationQuit()
    {
        StopListener();
    }

    public void StartListener()
    {
        if (running) return;

        try
        {
            udpClient = new UdpClient(listenPort);
            running = true;
            receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
            receiveThread.Start();
            Debug.Log($"UdpReceiver: Listening on port {listenPort}");
        }
        catch (Exception ex)
        {
            Debug.LogError("UdpReceiver: Failed to start listener: " + ex.Message);
        }
    }

    public void StopListener()
    {
        running = false;

        try
        {
            udpClient?.Close();
            udpClient = null;
        }
        catch { }

        try
        {
            if (receiveThread != null && receiveThread.IsAlive)
            {
                receiveThread.Join(500);
            }
        }
        catch { }

        receiveThread = null;
    }

    private void ReceiveLoop()
    {
        IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);

        while (running)
        {
            try
            {
                var data = udpClient.Receive(ref remoteEP); // blocking
                if (data != null && data.Length > 0)
                {
                    var msg = Encoding.UTF8.GetString(data);
                    lock (queueLock)
                    {
                        messageQueue.Enqueue(msg);
                    }
                }
            }
            catch (SocketException sex)
            {
                // Socket closed or interrupted
                if (running)
                {
                    Debug.LogWarning("UdpReceiver socket exception: " + sex.Message);
                }
                break;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("UdpReceiver receive error: " + ex.Message);
            }
        }
    }
}
