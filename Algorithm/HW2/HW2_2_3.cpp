
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
#include <iostream>
#include <string>

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
    int D[201][201]) // out : D 테이블
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
    const int D[201][201], // D 테이블
    string& editX,  // out : 편집 과정 X
    string& editY)  // out : 편집 과정 Y
{
    editX.clear();
    editY.clear();
    int i = n;
    int j = m;

    while (i > 0 || j > 0)
    {
        // X, Y 의 마지막 문자가 같으면 비용 0, 다르면 chg
        int c = 0;
        if (i > 0 && j > 0 && EQ(X[i - 1], Y[j - 1]))
            c = 0;
        else
            c = chg;

        // D[i][j]가 어떤 연산으로부터 왔는지 확인
        if (i > 0 && j > 0 && D[i][j] == D[i - 1][j - 1] + c) // 유지 또는 교체
        {
            editX += X[i - 1];
            editY += Y[j - 1];
            i--;
            j--;
        }
        else if (i > 0 && D[i][j] == D[i - 1][j] + del) // 삭제
        {
            editX += X[i - 1];
            editY += '-';
            i--;
        }
        else // 삽입
        {
            editX += '-';
            editY += Y[j - 1];
            j--;
        }
    }

    reverse(editX.begin(), editX.end());
    reverse(editY.begin(), editY.end());
}

// 각 열의 비용 합이 최소 비용과 같은지 검증
bool CalcCost( const string& X, const string& Y, // 원본 입력 문자열
                const string& editX, const string& editY, // 변경 입력 문자열
                int ins, int del, int chg, int& cost)
{
    cost = 0;
    if (editX.size() != editY.size())
        return false;

    string tmpX;
    string tmpY;

    for (size_t j = 0; j < editX.size(); j++)
    {
        // 두 줄이 모두 '-'인 열은 허용하지 않는다.
        if (editX[j] == '-' && editY[j] == '-')
            return false;

        if (editX[j] != '-')
            tmpX += editX[j];
        if (editY[j] != '-')
            tmpY += editY[j];

        if (editX[j] == '-')
            cost += ins;
        else if (editY[j] == '-')
            cost += del;
        else if (!EQ(editX[j], editY[j]))
            cost += chg;
    }

    return tmpX == X && tmpY == Y;
}

// 입력: 첫 줄 A, 둘째 줄 B, 셋째 줄 교체 비용 c. 빈 문자열은 빈 줄로 입력한다.
// 출력: 최소 비용, A의 편집 과정, B의 편집 과정을 각각 한 줄에 출력한다.
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string A;
    string B;
    int c;
    if (!getline(cin, A) || !getline(cin, B) || !(cin >> c))
    {
        cerr << "A, B, 정수 c를 순서대로 입력하세요.\n";
        return 1;
    }
    if (A.size() > 200 || B.size() > 200 || c < 1 || c > 100)
    {
        cerr << "A와 B의 길이는 0~200, c는 1~100이어야 합니다.\n";
        return 1;
    }
    if (A.find('-') != string::npos || B.find('-') != string::npos)
    {
        cerr << "'-'는 편집 과정의 공백 표시이므로 입력 문자열에 사용할 수 없습니다.\n";
        return 1;
    }

    // 전역 공유 변수
    const int ins = 1;
    const int del = 1;
    int n = static_cast<int>(A.size());
    int m = static_cast<int>(B.size());
    int D[201][201]; // D 테이블
    string editX; // X의 편집 과정
    string editY; // Y의 편집 과정

    // 최소 비용 계산 및 테이블 D 구축
    int cost = ED(A.c_str(), n, B.c_str(), m, ins, del, c, D);
    // 편집 과정 역추적
    TR(A.c_str(), n, B.c_str(), m, del, c, D, editX, editY);

    int editCost = 0;
    if (!CalcCost(A, B, editX, editY, ins, del, c, editCost))
    {
        cerr << "열별비용 합 검증 실패\n";
        return 1;
    }

    if (editCost != cost) {
        cerr << "편집 비용과 최소 비용 불일치\n";
        return 1;
    }

    cout << cost << '\n' << editX << '\n' << editY << '\n';
    return 0;
}
