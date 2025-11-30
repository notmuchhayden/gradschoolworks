using JetBrains.Annotations;
using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

// 자동차 및 사물 인식 인식 경계 박스
[Serializable]
public class Detection
{
    // 유니티 내부에서 사용하기 위한 감지 클래스
    public int classId;
    public float confidence;
    // 픽셀 단위의 경계 상자 좌표
    public float x1;
    public float y1;
    public float x2;
    public float y2;
}

// 차선 인식 직선 좌표 (픽셀 단위)
[Serializable]
public class LanePoint
{
    public float x1;
    public float y1;
    public float x2;
    public float y2;
}

// 차선 감지 결과 (왼쪽/오른쪽 차선)
[Serializable]
public class LaneDetection
{
    public LanePoint left_lane;  // 왼쪽 차선 좌표
    public LanePoint right_lane; // 오른쪽 차선 좌표
}

// 파이썬 JSON 전송 구조와 일치: {"type": "cars", "class": int, "confidence": float, "bbox": [x1,y1,x2,y2]}
[Serializable]
public class DetectionRaw
{
    // `class` 는 C# 키워드이므로, JsonUtility가 JSON 키를 매핑할 수 있도록 축자 식별자를 사용
    public string type;
    public int @class;
    public float confidence;
    public float[] bbox;
}

// 차선 인식 데이터 JSON 전송 구조: {"type": "lanes", "data": { "left_lane": {...}, "right_lane": {...} }}
[Serializable]
public class LaneDetectionRaw
{
    public string type;
    public LaneDetection data; // LaneDetection을 직접 재사용
}

[Serializable]
public class DetectionRawList
{
    public DetectionRaw[] detections;
}

public class UdpReceiver : MonoBehaviour
{
    [Tooltip("수신할 UDP 포트")]
    public int listenPort = 5005; // 파이썬 송신자 기본값과 일치

    // 최신 객체 감지 결과 저장
    public List<Detection> latestDetections = new List<Detection>();

    // 최신 차선 감지 결과 저장
    public LaneDetection latestLaneDetection = null;

    // 구독자가 사용할 수 있는 선택적 액션
    public event Action<List<Detection>> OnDetectionsReceived;
    public event Action<LaneDetection> OnLaneDetectionReceived;

    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool running;

    // 스레드 -> 메인 스레드 전달을 위한 내부 큐
    private readonly Queue<string> messageQueue = new Queue<string>();
    private readonly object queueLock = new object();

    void Start()
    {
        StartListener();
    }

    void Update()
    {
        // 메인 스레드에서 큐에 쌓인 메시지를 꺼내서 파싱
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
                ParseMessage(msg);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("UdpReceiver: 메시지 파싱 실패: " + ex.Message);
            }
        }
    }

    void ParseMessage(string msg)
    {
        string trimmed = msg.TrimStart();
        if (!trimmed.StartsWith("["))
        {
            Debug.LogWarning("UdpReceiver: JSON 배열이 예상되었지만 다음을 받았습니다: " + msg);
            return;
        }

        // 혼합 객체 배열로 파싱
        // Unity의 JsonUtility는 최상위 배열을 직접 파싱할 수 없으므로
        // 각 항목을 수동으로 추출하여 개별적으로 파싱합니다
        ParseMixedArray(msg);
    }

    void ParseMixedArray(string jsonArray)
    {
        // 외부 대괄호 제거 및 항목 파싱
        jsonArray = jsonArray.Trim();
        if (jsonArray.StartsWith("[")) jsonArray = jsonArray.Substring(1);
        if (jsonArray.EndsWith("]")) jsonArray = jsonArray.Substring(0, jsonArray.Length - 1);

        // 최상위 쉼표로 분할 (간단한 파서 - 잘 구성된 JSON을 가정)
        List<string> items = SplitJsonArray(jsonArray);

        latestDetections.Clear();
        latestLaneDetection = null;

        foreach (var item in items)
        {
            string itemTrimmed = item.Trim();
            if (string.IsNullOrEmpty(itemTrimmed)) continue;

            // type 필드 확인
            if (itemTrimmed.Contains("\"type\":\"cars\"") || itemTrimmed.Contains("\"type\": \"cars\""))
            {
                // 차량 감지로 파싱
                try
                {
                    var raw = JsonUtility.FromJson<DetectionRaw>(itemTrimmed);
                    if (raw != null)
                    {
                        var d = new Detection();
                        d.classId = raw.@class;
                        d.confidence = raw.confidence;
                        if (raw.bbox != null && raw.bbox.Length >= 4)
                        {
                            d.x1 = raw.bbox[0];
                            d.y1 = raw.bbox[1];
                            d.x2 = raw.bbox[2];
                            d.y2 = raw.bbox[3];
                        }
                        latestDetections.Add(d);
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("UdpReceiver: 차량 감지 파싱 실패: " + ex.Message);
                }
            }
            else if (itemTrimmed.Contains("\"type\":\"lanes\"") || itemTrimmed.Contains("\"type\": \"lanes\""))
            {
                // 차선 감지로 파싱
                try
                {
                    var raw = JsonUtility.FromJson<LaneDetectionRaw>(itemTrimmed);
                    if (raw != null && raw.data != null)
                    {
                        // 파싱된 LaneDetection을 직접 사용
                        latestLaneDetection = raw.data;
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogWarning("UdpReceiver: 차선 감지 파싱 실패: " + ex.Message);
                }
            }
        }

        // 결과 로그 출력
        if (latestDetections.Count > 0)
        {
            UnityEngine.Debug.Log($"UdpReceiver: {latestDetections.Count}개의 차량 감지를 받았습니다. 첫 번째 감지: " +
                $"classId={latestDetections[0].classId}, confidence={latestDetections[0].confidence}, " +
                $"bbox=({latestDetections[0].x1}, {latestDetections[0].y1}, {latestDetections[0].x2}, {latestDetections[0].y2})");
        }

        if (latestLaneDetection != null)
        {
            string leftInfo = latestLaneDetection.left_lane != null ? 
                $"[{latestLaneDetection.left_lane.x1}, {latestLaneDetection.left_lane.y1}, {latestLaneDetection.left_lane.x2}, {latestLaneDetection.left_lane.y2}]" : "null";
            string rightInfo = latestLaneDetection.right_lane != null ? 
                $"[{latestLaneDetection.right_lane.x1}, {latestLaneDetection.right_lane.y1}, {latestLaneDetection.right_lane.x2}, {latestLaneDetection.right_lane.y2}]" : "null";
            UnityEngine.Debug.Log($"UdpReceiver: 차선 감지를 받았습니다. 왼쪽: {leftInfo}, 오른쪽: {rightInfo}");
        }

        // 이벤트 호출
        OnDetectionsReceived?.Invoke(latestDetections);
        if (latestLaneDetection != null)
        {
            OnLaneDetectionReceived?.Invoke(latestLaneDetection);
        }
    }

    // 간단한 JSON 배열 분할기 - 깊이 0에서 쉼표로 분할
    List<string> SplitJsonArray(string content)
    {
        List<string> items = new List<string>();
        int depth = 0;
        int start = 0;

        for (int i = 0; i < content.Length; i++)
        {
            char c = content[i];
            
            if (c == '{' || c == '[')
            {
                depth++;
            }
            else if (c == '}' || c == ']')
            {
                depth--;
            }
            else if (c == ',' && depth == 0)
            {
                items.Add(content.Substring(start, i - start));
                start = i + 1;
            }
        }

        // 마지막 항목 추가
        if (start < content.Length)
        {
            items.Add(content.Substring(start));
        }

        return items;
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
            Debug.Log($"UdpReceiver: 포트 {listenPort}에서 수신 대기 중");
        }
        catch (Exception ex)
        {
            Debug.LogError("UdpReceiver: 리스너 시작 실패: " + ex.Message);
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
                var data = udpClient.Receive(ref remoteEP); // 블로킹
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
                // 소켓이 닫히거나 중단됨
                if (running)
                {
                    Debug.LogWarning("UdpReceiver 소켓 예외: " + sex.Message);
                }
                break;
            }
            catch (Exception ex)
            {
                Debug.LogWarning("UdpReceiver 수신 오류: " + ex.Message);
            }
        }
    }
}
