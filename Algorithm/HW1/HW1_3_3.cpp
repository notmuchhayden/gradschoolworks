#include <iostream>
#include <vector>

using namespace std;

// 강의록의 MergeSort 코드를 이용하여 역전쌍을 세는 프로그램

// 역전쌍의 총 개수
long long InversionCount = 0;

// 역전쌍의 개수 한번에 더하기 :  Len(L) - 1 
void CountInversion(int i, int n)
{
    // B[i], B[i+1], ..., B[n-1] 모두 C[j]보다 크므로
    // 총 n-i개의 역전쌍이 발생한다.
    InversionCount += (n - i);
}

// 합병
vector<long long> Merge(vector<long long> B, vector<long long> C, int n, int m)
{
    vector<long long> A(n + m);

    int i = 0;
    int j = 0;
    int k = 0;

    // 두 배열을 앞에서부터 비교하면서 합병
    while (i < n && j < m)
    {
        if (B[i] <= C[j])
        {
            A[k++] = B[i++];
        }
        else
        {
            // B[i] > C[j] 일 때, B는 이미 정렬된 상태이므로 
            // B[i] 의 뒷부분은 모두 C[j]보다 크다.
            // 따라서 n-i개의 역전쌍을 한 번에 추가한다.
            CountInversion(i, n);

            A[k++] = C[j++];
        }
    }

    // B에 남은원소 이동
    for (; i < n; i++)
    {
        A[k++] = B[i];
    }

    // C에 남은원소 이동
    for (; j < m; j++)
    {
        A[k++] = C[j];
    }

    return A;
}


// 합병정렬
vector<long long> MergeSort(vector<long long> A, int n)
{
    if (n > 1)
    {
        int Mid = n / 2;
        // 분할
        vector<long long> Left(A.begin(), A.begin() + Mid);// 왼쪽 부분배열
        vector<long long> Right(A.begin() + Mid, A.end());// 오른쪽 부분배열

        // 정복
        vector<long long> B = MergeSort(Left, Mid);
        vector<long long> C = MergeSort(Right, n - Mid);

        // 결합
        A = Merge(B, C, Mid, n - Mid);
    }

    return A;
}


int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n; // 첫째 입력

    vector<long long> A(n);

    for (int i = 0; i < n; i++)
    {
        cin >> A[i]; // 둘째 입력
    }

    MergeSort(A, n);

    cout << InversionCount << '\n'; // 출력 : 역전쌍의 총 개수

    return 0;
}