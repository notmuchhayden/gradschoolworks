using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ADASCarController : MonoBehaviour
{
    [SerializeField]
    Camera mainCamera;
    [SerializeField]
    GameObject otherCar;

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

    // pool storage
    List<GameObject> pool;
    Transform poolParent;

    // Start is called before the first frame update
    void Start()
    {
        udpReceiver = GetComponent<UdpReceiver>();

        // create pool parent to keep hierarchy clean
        poolParent = new GameObject("OtherCarPool").transform;
        poolParent.parent = this.transform;

        // initialize pool
        pool = new List<GameObject>(poolSize);
        if (otherCar != null)
        {
            for (int i = 0; i < poolSize; i++)
            {
                var go = Instantiate(otherCar, Vector3.zero, Quaternion.identity, poolParent);
                go.SetActive(false); // invisible until used
                pool.Add(go);
            }
        }
        else
        {
            Debug.LogWarning("ADASCarController: otherCar prefab is not assigned. Pool will be empty.");
        }
    }

    // Update is called once per frame
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

        // 최신 탐지 결과를 가져옵니다.
        var latestDetections = udpReceiver.latestDetections;

        if (latestDetections == null || latestDetections.Count == 0)
        {
            // deactivate all pooled objects
            for (int i = 0; i < pool.Count; i++)
            {
                if (pool[i] != null && pool[i].activeSelf)
                    pool[i].SetActive(false);
            }
            return;
        }

        int used = Mathf.Min(poolSize, latestDetections.Count);

        // Update used pooled objects with detection positions
        for (int i = 0; i < used; i++)
        {
            var det = latestDetections[i];

            // bounding box 중심점 계산 (YOLO bbox: x1,y1,x2,y2 in pixels)
            float cx = (det.x1 + det.x2) * 0.5f;
            float cy = (det.y1 + det.y2) * 0.5f;

            // 변환: 영상 원점이 좌상단이라고 가정
            float vx = cx / (float)videoWidth;              // 0..1 (left..right)
            float vy = 1f - (cy / (float)videoHeight);      // 0..1 (bottom..top)

            // 카메라 뷰포트에서 레이 생성
            Ray ray = mainCamera.ViewportPointToRay(new Vector3(vx, vy, 0f));

            Vector3 worldPos = Vector3.zero;
            bool hitPlane = false;

            // XZ 평면(y=0)과의 교차 계산
            if (Mathf.Abs(ray.direction.y) > 1e-6f)
            {
                float t = (0f - ray.origin.y) / ray.direction.y;
                if (t > 0f)
                {
                    worldPos = ray.origin + ray.direction * t;
                    hitPlane = true;
                }
            }


            var go = pool[i];
            if (go == null) continue;

            if (hitPlane)
            {
                go.transform.position = worldPos;
                if (!go.activeSelf)
                    go.SetActive(true);

                // optional: destroy after lifetime by scheduling deactivation instead of destroy
                if (instantiateLifetime > 0f)
                {
                    // cancel any previous invocation and schedule deactivate
                    CancelInvoke("DeactivatePoolItem");
                    Invoke("DeactivatePoolItem", instantiateLifetime);
                }
            }
            else
            {
                if (go.activeSelf)
                    go.SetActive(false);
            }
        }

        // Deactivate unused pool items
        for (int i = used; i < pool.Count; i++)
        {
            var go = pool[i];
            if (go != null && go.activeSelf)
                go.SetActive(false);
        }
    }

    // deactivates all pool items (invoked after lifetime)
    void DeactivatePoolItem()
    {
        for (int i = 0; i < pool.Count; i++)
        {
            var go = pool[i];
            if (go != null && go.activeSelf)
                go.SetActive(false);
        }
    }
}
