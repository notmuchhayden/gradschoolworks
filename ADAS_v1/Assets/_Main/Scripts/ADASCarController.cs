using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ADASCarController : MonoBehaviour
{
    [SerializeField]
    Camera mainCamera;
    [SerializeField]
    GameObject otherCar;

    // 차선 표시용 게임 오브젝트
    [SerializeField]
    GameObject leftLane;
    [SerializeField]
    GameObject rightLane;

    // UDP 수신기
    UdpReceiver udpReceiver;

    [Header("Source video size (pixels)")]
    [SerializeField]
    int videoWidth = 1280;
    [SerializeField]
    int videoHeight = 720;

    [Header("Instantiated marker lifetime (seconds)")]
    [SerializeField]
    float instantiateLifetime = 5f;

    [Header("Pooling")]
    [SerializeField]
    int poolSize = 20;

    [Header("Depth scaling by bbox area")]
    [Tooltip("worldPos.z에 적용되는 스케일. 바운딩 박스 면적이 작을수록 멀리(더 큰 스케일), 면적이 클수록 가까이(더 작은 스케일)로 반영.")]
    [SerializeField]
    float minScale = 0.5f;
    [SerializeField]
    float maxScale = 2.0f;
    [Tooltip("정규화된 바운딩 박스 면적에 적용되는 민감도 배수(값이 클수록 minScale로 더 빨리 이동).")]
    [SerializeField]
    float areaSensitivity = 10f;

    [Header("Exclusion")]
    [Tooltip("XZ 좌표에서 원점으로부터의 거리가 이 값 이하이면 풀 할당에서 해당 탐지를 무시")]
    [SerializeField]
    float excludeDistance = 1.5f;

    // 속도를 위해 풀을 미리 생성
    List<GameObject> pool;
    Transform poolParent;

    void Start()
    {
        udpReceiver = GetComponent<UdpReceiver>();

        // 계층 구조를 깔끔하게 유지하기 위해 풀의 부모 객체를 생성
        poolParent = new GameObject("OtherCarPool").transform;
        poolParent.parent = this.transform;

        // 풀을 초기화
        pool = new List<GameObject>(poolSize);
        if (otherCar != null)
        {
            for (int i = 0; i < poolSize; i++)
            {
                var go = Instantiate(otherCar, Vector3.zero, Quaternion.identity, poolParent);
                go.SetActive(false); // 사용될 때까지 비활성화(보이지 않음)
                pool.Add(go);
            }
        }
        else
        {
            Debug.LogWarning("ADASCarController: otherCar prefab is not assigned. Pool will be empty.");
        }
    }

    // Update는 매 프레임 한 번 호출
    void Update()
    {
        if (udpReceiver == null || mainCamera == null || pool == null)
        {
            if (mainCamera == null)
            {
                Debug.LogWarning("ADASCarController: mainCamera is not assigned.");
            }
            if (otherCar == null)
            {
                Debug.LogWarning("ADASCarController: otherCar (prefab) is not assigned.");
            }
            return;
        }

        // 최신 탐지 결과를 가져온다
        var latestDetections = udpReceiver.latestDetections;

        if (latestDetections == null || latestDetections.Count == 0)
        {
            // 풀에 있는 모든 객체를 비활성화
            for (int i = 0; i < pool.Count; i++)
            {
                if (pool[i] != null && pool[i].activeSelf)
                    pool[i].SetActive(false);
            }

            // 차선 정보가 없을 때도 차선 오브젝트를 숨기지 않고 위치 업데이트만 건너뜀
            UpdateLanesFromUdp(); // 최신LaneDetection이 있을 수 있으므로 시도
            return;
        }

        int poolIndex = 0;

        // 탐지 항목을 반복하여 순차적으로 풀에 할당하되, 제외 조건에 해당하는 것은 건너뜀
        for (int di = 0; di < latestDetections.Count && poolIndex < poolSize; di++)
        {
            var det = latestDetections[di];

            // 경계 상자 중심점 계산 (YOLO bbox: x1,y1,x2,y2, 픽셀 단위)
            float cx = (det.x1 + det.x2) * 0.5f;
            float cy = (det.y1 + det.y2) * 0.5f;

            // 영상 원점은 좌상단
            float vx = cx / (float)videoWidth;
            float vy = 1f - (cy / (float)videoHeight);

            // 카메라 뷰포트에서 레이를 생성
            Ray ray = mainCamera.ViewportPointToRay(new Vector3(vx, vy, 0f));

            Vector3 worldPos = Vector3.zero;
            bool hitPlane = false;

            // XZ 평면(y=0)과의 교차를 계산합
            if (Mathf.Abs(ray.direction.y) > 1e-6f)
            {
                float t = (0f - ray.origin.y) / ray.direction.y;
                if (t > 0f)
                {
                    worldPos = ray.origin + ray.direction * t;
                    hitPlane = true;
                }
            }

            if (!hitPlane)
                continue;

            // 바운딩 박스 면적과 정규화된 비율을 계산. 박스 면적이 작을수록 멀리 있는 것으로 간주.
            float bw = Mathf.Abs(det.x2 - det.x1);
            float bh = Mathf.Abs(det.y2 - det.y1);
            float area = bw * bh;
            float videoArea = Mathf.Max(1, videoWidth * videoHeight);
            float areaRatio = area / videoArea;

            // 면적 비율을 스케일로 매핑: 작은 면적 -> maxScale(멀리), 큰 면적 -> minScale(가까이)
            float mapped = Mathf.Clamp01(areaRatio * areaSensitivity);
            float scale = Mathf.Lerp(maxScale, minScale, mapped);

            // z 좌표에 스케일을 적용합니다.
            worldPos.z = worldPos.z * scale;

            // 제외 규칙: XZ 원점으로부터의 거리가 excludeDistance 이하이면 이 탐지를 건너뜀
            Vector2 xz = new Vector2(worldPos.x, worldPos.z);
            if (xz.magnitude <= excludeDistance)
            {
                continue;
            }

            var go = pool[poolIndex];
            if (go == null)
            {
                poolIndex++;
                continue;
            }

            worldPos.y = 0.5f; // 약간 띄워서 표시
            go.transform.position = worldPos;

            if (!go.activeSelf)
                go.SetActive(true);

            poolIndex++;
        }

        // 사용되지 않은 풀 항목들을 비활성화합니다.
        for (int i = poolIndex; i < pool.Count; i++)
        {
            var go = pool[i];
            if (go != null && go.activeSelf)
                go.SetActive(false);
        }

        // 차선 정보를 사용해 leftLane, rightLane 게임 오브젝트의 x 좌표를 결정
        UpdateLanesFromUdp();

        // 선택사항: lifetime 후에 전역 비활성화를 예약
        if (instantiateLifetime > 0f)
        {
            CancelInvoke("DeactivatePoolItem");
            Invoke("DeactivatePoolItem", instantiateLifetime);
        }
    }

    void DeactivatePoolItem()
    {
        for (int i = 0; i < pool.Count; i++)
        {
            var go = pool[i];
            if (go != null && go.activeSelf)
                go.SetActive(false);
        }
    }

    // 차선 위치 계산
    void UpdateLanesFromUdp()
    {
        if (udpReceiver == null || mainCamera == null) return;

        var lanes = udpReceiver.latestLaneDetection;
        if (lanes == null) return;

        // 헬퍼 로컬 함수: LanePoint -> world intersection x (nullable)
        float? GetLaneWorldX(LanePoint lp)
        {
            if (lp == null) return null;

            // 픽셀 좌표 (원점: 좌상단) -> 뷰포트 좌표 (0..1, y: 아래0 위1)
            float vx = lp.x1 / (float)videoWidth;
            float vy = 1f - (lp.y1 / (float)videoHeight);

            Ray ray = mainCamera.ViewportPointToRay(new Vector3(vx, vy, 0f));
            if (Mathf.Abs(ray.direction.y) <= 1e-6f) 
                return null;

            float t = (0f - ray.origin.y) / ray.direction.y;
            if (t <= 0f) 
                return null;

            Vector3 wp = ray.origin + ray.direction * t;
            return wp.x;
        }

        // 왼쪽 차선
        if (leftLane != null && lanes.left_lane != null)
        {
            float? lx = GetLaneWorldX(lanes.left_lane);
            if (lx.HasValue)
            {
                Vector3 p = leftLane.transform.position;
                p.x = lx.Value;
                leftLane.transform.position = p;
            }
        }

        // 오른쪽 차선
        if (rightLane != null && lanes.right_lane != null)
        {
            float? rx = GetLaneWorldX(lanes.right_lane);
            if (rx.HasValue)
            {
                Vector3 p = rightLane.transform.position;
                p.x = rx.Value;
                rightLane.transform.position = p;
            }
        }
    }
}
