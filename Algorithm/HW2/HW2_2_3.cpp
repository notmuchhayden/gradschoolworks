
/*
(3) 임의의 A, B, c를 받아 최소 비용과 최적 편집 과정 하나를 출력하는 프로그램을 구현
하세요. 길이는 각각 0~200, c는 1~100의 정수입니다. 아래 5쌍을 c=1과 c=3에서 각각
실행하고 결과를 표로 정리하세요. 시간 및 공간 복잡도도 설명하세요.

+------------------+-----------+-----------+
| 검증 항목        |     A     |      B    |
|------------------+-----------+-----------|
| 수기 결과 대조   | abc       | Yabd      |
| 삽입만 필요      | 빈 문자열 | Abc       |
| 삭제만 필요      | abc       | 빈 문자열 |
| 변경 불필요      | same      | Same      |
| 순서가 다른 경우 | ab        | Ba        |
+------------------+-----------+-----------+

편집 과정은 -를 삽입하여 길이를 맞춘 두 문자열로 출력해도 됩니다. 각 열은 문자 유지·
교체·삽입·삭제 중 하나를 나타냅니다. 공백 자리는 "-"로 표시합니다. 예를 들어 "a-"와
"ab"는 a를 유지하고 b를 삽입한 결과입니다. 두 줄에서 "-"를 제거하면 각각 A와 B가 되
어야 하며, 두 줄이 모두 "-"인 열은 허용하지 않습니다. 각 열의 비용 합이 계산한 최소
비용과 같은지도 확인하세요. 최적 편집 과정이 여러 개이면 그중 하나를 출력하면 되며,
동률 처리 순서는 자유입니다.

*/

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

// 문제에서 대소문자 구분이 없으므로 EQ()를 사용하여 비교.
bool EQ(char X, char Y)
{
    return tolower(static_cast<unsigned char>(X))
        == tolower(static_cast<unsigned char>(Y));
}

// '[알고리즘 특론 4강] 동적프로그래밍 방법.pdf' 에서 
// 설명한 편집 거리 알고리즘을 구현
int ED(
    const char X[], // 입력: X[0..n-1], 
    int n,          // 문자배열 X의 크기
    const char Y[], // 입력: Y[0..m-1]
    int m,          // 문자배열 Y의 크기
    int ins,        // 삽입 비용
    int del,        // 삭제 비용
    int chg,        // 변경 비용
    vector<vector<int>>& D) // D 테이블
{
    D[0][0] = 0;

    
    for (int i = 1; i < n + 1; i++)     // 첫 열의 초기화
        D[i][0] = D[i - 1][0] + del;
    for (int j = 1; j < m + 1; j++)     // 첫 행의 초기화
        D[0][j] = D[0][j - 1] + ins;

    for (int i = 1; i < n + 1; i++)
    {
        for (int j = 1; j < m + 1; j++)
        {
            int c = EQ(X[i - 1], Y[j - 1]) ? 0 : chg;
            D[i][j] = min({
                D[i - 1][j] + del,
                D[i][j - 1] + ins,
                D[i - 1][j - 1] + c
            });
        }
    }

    return D[n][m];
}

// 편집 과정 역추적
void TR(
    const char X[], // 입력: X[0..n-1],
    int n,          // 문자배열 X의 크기
    const char Y[], // 입력: Y[0..m-1]
    int m,          // 문자배열 Y의 크기
    int del,        // 삭제 비용
    int chg,        // 변경 비용
    const vector<vector<int>>& D, // D 테이블
    string& editX,  // 출력: 편집 과정 X
    string& editY)  // 출력: 편집 과정 Y
{
    editX.clear();
    editY.clear();
    int i = n;
    int j = m;

    while (i > 0 || j > 0)
    {
        int c = (i > 0 && j > 0 && EQ(X[i - 1], Y[j - 1]))
            ? 0 : chg;

        if (i > 0 && j > 0 && D[i][j] == D[i - 1][j - 1] + c)
        {
            editX += X[i - 1];
            editY += Y[j - 1];
            i--;
            j--;
        }
        else if (i > 0 && D[i][j] == D[i - 1][j] + del)
        {
            editX += X[i - 1];
            editY += '-';
            i--;
        }
        else
        {
            editX += '-';
            editY += Y[j - 1];
            j--;
        }
    }

    reverse(editX.begin(), editX.end());
    reverse(editY.begin(), editY.end());
}

// 정렬된 두 줄에서 원래 문자열을 복원하고 열별 비용을 합산한다.
// 두 줄이 모두 '-'인 열은 허용하지 않는다.
// 문제의 표현 방식에서 '-'는 공백 표시용 문자이다.
bool CheckEdit(const string& A, const string& B,
               const string& editX, const string& editY,
               int ins, int del, int chg, int& cost)
{
    cost = 0;
    if (editX.size() != editY.size())
        return false;

    string X;
    string Y;

    for (size_t j = 0; j < editX.size(); j++)
    {
        if (editX[j] == '-' && editY[j] == '-')
            return false;

        if (editX[j] != '-')
            X += editX[j];
        if (editY[j] != '-')
            Y += editY[j];

        if (editX[j] == '-')
            cost += ins;
        else if (editY[j] == '-')
            cost += del;
        else if (!EQ(editX[j], editY[j]))
            cost += chg;
    }

    return X == A && Y == B;
}

string Display(const string& X)
{
    return X.empty() ? "(empty)" : X;
}

// 문제에 지정된 5쌍을 c = 1, c = 3에서 각각 실행하여 표로 출력한다.
bool PrintExamples()
{
    const vector<pair<string, string>> examples = {
        {"abc", "Yabd"},
        {"", "Abc"},
        {"abc", ""},
        {"same", "Same"},
        {"ab", "Ba"}
    };
    const int ins = 1;
    const int del = 1;
    bool allValid = true;

    cout << "삽입/삭제 비용 = 1, 대소문자 구분 없음\n"
         << "(empty)는 빈 문자열, '-'는 편집 과정의 공백이다.\n\n";
    cout << left
         << setw(10) << "A" << setw(10) << "B"
         << setw(5) << "c" << setw(8) << "Minimum"
         << setw(12) << "Edit A" << setw(12) << "Edit B"
         << setw(8) << "Sum" << "Check\n";
    cout << string(70, '-') << '\n';

    for (const auto& example : examples)
    {
        for (int c : {1, 3})
        {
            const string& A = example.first;
            const string& B = example.second;
            int n = static_cast<int>(A.size());
            int m = static_cast<int>(B.size());
            vector<vector<int>> D(n + 1, vector<int>(m + 1));
            string editX;
            string editY;

            int cost = ED(A.c_str(), n, B.c_str(), m, ins, del, c, D);
            TR(A.c_str(), n, B.c_str(), m, del, c, D, editX, editY);
            int editCost = 0;
            bool valid = CheckEdit(A, B, editX, editY, ins, del, c, editCost)
                && editCost == cost;
            allValid = allValid && valid;

            cout << setw(10) << Display(A) << setw(10) << Display(B)
                 << setw(5) << c << setw(8) << cost
                 << setw(12) << Display(editX) << setw(12) << Display(editY)
                 << setw(8) << editCost << (valid ? "OK" : "FAIL") << '\n';
        }
    }

    return allValid;
}

// 실행하면 지정된 5쌍을 c = 1, c = 3에서 계산한 결과표를 출력한다.
int main()
{
    ios::sync_with_stdio(false);

    return PrintExamples() ? 0 : 1;
}
