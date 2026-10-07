
/*
(2) 배열을 이용하여 선형 탐사와 이중 해싱의 삽입·탐색을 직접 구현하고, 제공된 입력 파
일로 성능을 비교하세요. 탐사 위치는 다음과 같습니다. (코딩, 10점)
- 선형 탐사 : H(k, i) = ((k mod m) + i) mod m
- 이중 해싱 :
	- H(k, i) = ((k mod m) + i * h2(k)) mod m
	- h2(k) = 1 + (k mod (m - 1))
	
[구현 조건]
키는 중복 없는 0 이상의 정수이며, 빈칸은 키 0 과 구분하세요. 탐색은 키를 찾거나 빈칸을
만나면 종료하고, 최대 m회까지 탐사합니다. 실험 중 삭제와 테이블 크기 변경은 하지 않
습니다. 핵심 연산을 사전·집합의 내장 기능으로 대신하지 마세요.

[실험 입력]
테이블 크기는 m=1009 입니다. 다음 네 파일을 두 방식에 각각 적용하여 총 8개의 조건을
실험하세요. 파일 형식과 입력 생성 규칙은 README.txt에 제시되어 있습니다.

+--------------------+-----------+------------+
| 입력 유형          | n=200     | n=800      |
+--------------------+-----------+------------+ 
| R (무작위 입력)    | R_200.txt | R_800.txt  |
| R (충돌 집중 입력) | C_200.txt | C_800.txt  |
+--------------------+-----------+------------+ 

매 실험은 빈 테이블에서 시작합니다. 파일에 주어진 순서대로 키 n개를 모두 삽입한 뒤,
테이블을 변경하지 않고 저장된 키 n개와 실패 탐색 질의 n개를 각각 한 번씩 탐색하세요.
두 방식에 동일한 삽입·질의 순서를 사용합니다.

[결과 정리]
● 실험 결과 포함 항목 : 각 조건에 대한 삽입 평균 탐사 횟수, 성공 탐색의 평균 탐사 횟
수, 실패 탐색의 평균 및 최대 탐사 횟수. 결과 재현에 필요한 실제 사용 키, 질의 순서
도 필수

*/

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;

const int M = 1009;
const int EMPTY = -1; // 키는 0 이상이므로 빈칸을 -1로 구분한다.

// 원문 주석의 'm mod m'은 오기로 보고, 강의록의 h1(k) = k mod m을 사용한다.
// doubleHash가 false이면 선형 탐사, true이면 이중 해싱이다.
int H(int k, int i, bool doubleHash)
{
    int step = 1;
    if (doubleHash)
        step = 1 + k % (M - 1);

    return (k % M + i * step) % M;
}

// 빈칸을 확인한 탐사도 포함하여 탐사 횟수를 반환한다. 삽입 실패는 -1이다.
int Insert(int T[], int k, bool doubleHash)
{
    for (int i = 0; i < M; i++)
    {
        int j = H(k, i, doubleHash);
        if (T[j] == EMPTY)
        {
            T[j] = k;
            return i + 1;
        }
    }
    return -1;
}

// 찾은 위치를 반환하며, 찾지 못하면 -1을 반환한다. count는 탐사 횟수이다.
int Search(const int T[], int k, bool doubleHash, int& count)
{
    count = 0;
    for (int i = 0; i < M; i++)
    {
        int j = H(k, i, doubleHash);
        count++;
        if (T[j] == k)
            return j;
        if (T[j] == EMPTY)
            return -1;
    }
    return -1;
}

// 하나의 방식으로 삽입, 성공 탐색, 실패 탐색을 수행한다.
bool Experiment(const int keys[], const int queries[], int n, bool doubleHash, ostream& output)
{
    int T[M];
    for (int i = 0; i < M; i++)
        T[i] = EMPTY;

    int insertSum = 0;
    int successSum = 0;
    int failSum = 0;
    int failMax = 0;

    for (int i = 0; i < n; i++)
    {
        int count = Insert(T, keys[i], doubleHash);
        if (count == -1)
            return false;
        insertSum += count;
    }

    for (int i = 0; i < n; i++)
    {
        int count;
        if (Search(T, keys[i], doubleHash, count) == -1)
            return false;
        successSum += count;
    }

    for (int i = 0; i < n; i++)
    {
        int count;
        if (Search(T, queries[i], doubleHash, count) != -1)
            return false;
        failSum += count;
        if (count > failMax)
            failMax = count;
    }

    if (doubleHash)
        output << "이중 해싱";
    else
        output << "선형 탐사";

    output << " | " << (double)insertSum / n
         << " | " << (double)successSum / n
         << " | " << (double)failSum / n
         << " | " << failMax << '\n';
    return true;
}

int main()
{
#ifdef _WIN32
    // /utf-8로 빌드한 한글 문자열에 맞춰 콘솔 출력 인코딩을 설정한다.
    SetConsoleOutputCP(CP_UTF8);
#endif

    ofstream result("result.txt");
    if (!result)
    {
        cerr << "result.txt 파일을 생성할 수 없습니다.\n";
        return 1;
    }

    // 콘솔과 파일에 같은 결과를 출력하도록 전체 출력을 모은다.
    ostringstream output;
    const char* files[4] = { "R_200.txt", "R_800.txt", "C_200.txt", "C_800.txt" };
    output << fixed << setprecision(5);

    // 입력 파일이 있는 Algorithm/HW2 폴더에서 실행한다.
    for (int f = 0; f < 4; f++)
    {
        ifstream input(files[f]);
        if (!input)
        {
            cerr << files[f] << " 파일을 열 수 없습니다. 입력 파일이 있는 폴더에서 실행하세요.\n";
            return 1;
        }

        int m, n;
        if (!(input >> m >> n) || m != M || n < 1 || n > M)
        {
            cerr << files[f] << "의 m, n 값이 올바르지 않습니다.\n";
            return 1;
        }

        int keys[M];
        int queries[M];
        for (int i = 0; i < n; i++)
        {
            if (!(input >> keys[i]) || keys[i] < 0)
            {
                cerr << files[f] << "의 삽입 키를 읽을 수 없습니다.\n";
                return 1;
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (!(input >> queries[i]) || queries[i] < 0)
            {
                cerr << files[f] << "의 실패 질의를 읽을 수 없습니다.\n";
                return 1;
            }
        }

        // 두 방식 모두 아래에 출력한 순서를 그대로 사용한다.
        output << '\n' << files[f] << " (m=" << m << ", n=" << n << ")\n";
        output << "삽입 키 및 성공 탐색 순서:\n";
        for (int i = 0; i < n; i++)
            output << keys[i] << (i == n - 1 ? '\n' : ' ');
        output << "실패 탐색 질의 순서:\n";
        for (int i = 0; i < n; i++)
            output << queries[i] << (i == n - 1 ? '\n' : ' ');

        output << "방식 | 삽입 평균 | 성공 탐색 평균 | 실패 탐색 평균 | 실패 탐색 최대\n";
        if (!Experiment(keys, queries, n, false, output) || !Experiment(keys, queries, n, true, output))
        {
            cerr << "삽입 또는 탐색 결과가 예상과 다릅니다.\n";
            return 1;
        }
    }
    cout << output.str();
    result << output.str();
    result.close();
    if (!result)
    {
        cerr << "result.txt 파일에 결과를 저장할 수 없습니다.\n";
        return 1;
    }
    return 0;
}
