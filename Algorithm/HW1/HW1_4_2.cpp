#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <limits>

using namespace std;

struct Counselor
{
    string ID;
    long long StartTime;
    long long EndTime;
};

// 정렬
bool Compare(const Counselor& a, const Counselor& b)
{
    // 1. 완료시간
    if (a.EndTime != b.EndTime)
        return a.EndTime < b.EndTime;

    // 2. 완료시간이 같으면 시작시간
    if (a.StartTime != b.StartTime)
        return a.StartTime < b.StartTime;

    // 3. 시작시간도 같으면 상담 ID 순
    return a.ID < b.ID;
}

// 상담 선택
vector<string> ActivitySelection(vector<Counselor>& T)
{
    // 완료시간 -> 시작시간 -> ID 순으로 정렬
    sort(T.begin(), T.end(), Compare);

    vector<string> Selected;

    // 아직 아무 상담도 선택하지 않은 상태
    long long LastEndTime = numeric_limits<long long>::min();

    for (int i = 0; i < (int)T.size(); i++)
    {
        // 앞 상담의 완료시간과 다음 상담의 시작시간이
        // 같은 경우도 선택 가능하므로 >= 사용
        if (T[i].StartTime >= LastEndTime)
        {
            Selected.push_back(T[i].ID);
            LastEndTime = T[i].EndTime;
        }
    }

    return Selected;
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<Counselor> T(n);

    // n개의 상담 입력
    for (int i = 0; i < n; i++)
    {
        cin >> T[i].ID
            >> T[i].StartTime
            >> T[i].EndTime;
    }

    vector<string> Selected = ActivitySelection(T);

    // 첫째 줄: 최대 상담 수
    cout << Selected.size() << '\n';

    // 둘째 줄: 선택된 상담 ID
    for (int i = 0; i < (int)Selected.size(); i++)
    {
        if (i > 0)
            cout << ' ';

        cout << Selected[i];
    }

    cout << '\n';

    return 0;
}