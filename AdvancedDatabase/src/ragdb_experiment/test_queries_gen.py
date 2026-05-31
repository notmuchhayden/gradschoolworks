# 테스트 쿼리 생성 스크립트

from pathlib import Path
import json

QUERY_TEMPLATES = {
    "World": [
        "international politics and world conflict",
        "global diplomacy and foreign policy",
        "war peace negotiations and international relations",
        "elections government and political leaders worldwide",
        "United Nations summit and global security",
        "middle east conflict and peace talks",
        "European politics and diplomatic relations",
        "Asian countries political crisis and reforms",
        "international sanctions and government response",
        "world leaders meeting and treaty agreement",
        "border dispute and military tension",
        "humanitarian crisis refugees and international aid",
        "terrorism security alert and global response",
        "presidential election and foreign affairs",
        "diplomatic talks between countries",
        "global protest movement and government policy",
        "international law and human rights issue",
        "military operation and regional security",
        "peace agreement and ceasefire negotiations",
        "foreign minister visit and bilateral relations",
        "political scandal and national government",
        "world news about crisis and diplomacy",
        "global organization response to conflict",
        "country leadership change and election result",
        "international relations and geopolitical strategy",
    ],
    "Sports": [
        "football baseball basketball sports match",
        "soccer team wins championship game",
        "baseball season playoff and league standings",
        "basketball player scores in final match",
        "tennis tournament champion and grand slam",
        "Olympic athletes medals and competition results",
        "golf tournament final round and winner",
        "football coach strategy and team performance",
        "sports injury update and player recovery",
        "world cup match and national team",
        "basketball playoffs and team victory",
        "baseball pitcher performance and home run",
        "soccer transfer news and club contract",
        "hockey game score and league result",
        "racing event driver wins title",
        "boxing match champion and fight result",
        "college sports team and season opener",
        "athlete retirement and career record",
        "sports league schedule and final standings",
        "championship final game and winning team",
        "training camp player roster and coach",
        "stadium crowd and major sports event",
        "tennis player ranking and tournament match",
        "basketball draft prospect and team selection",
        "baseball league trade and team lineup",
    ],
    "Business": [
        "stock market company earnings business economy",
        "corporate merger acquisition and investor reaction",
        "interest rates inflation and central bank policy",
        "technology company quarterly earnings report",
        "oil prices energy market and global economy",
        "retail sales consumer spending and economic outlook",
        "startup funding venture capital investment",
        "banking sector profit and financial regulation",
        "stock exchange trading and market volatility",
        "company bankruptcy restructuring and debt",
        "automaker sales revenue and production forecast",
        "airline business travel demand and profit",
        "housing market mortgage rates and real estate",
        "currency exchange rates and global trade",
        "job market unemployment and wage growth",
        "business deal partnership and expansion plan",
        "technology shares rise after earnings",
        "economic growth forecast and recession risk",
        "consumer confidence and market report",
        "supply chain disruption and manufacturing cost",
        "pharmaceutical company revenue and product sales",
        "financial market analysts and investment outlook",
        "corporate leadership change and business strategy",
        "trade deficit exports imports and economy",
        "company stock falls after weak guidance",
    ],
    "Sci/Tech": [
        "technology software internet computer science",
        "new smartphone launch and mobile technology",
        "space mission satellite and rocket launch",
        "artificial intelligence research and machine learning",
        "software security vulnerability and patch update",
        "internet company product and online service",
        "computer chip semiconductor and processor design",
        "science discovery researchers and laboratory study",
        "medical technology innovation and biotech research",
        "NASA space exploration and Mars mission",
        "cybersecurity attack data breach and protection",
        "robotics automation and industrial technology",
        "wireless network broadband and telecom service",
        "open source software release and developer tools",
        "climate science study and environmental technology",
        "gaming console hardware and video game industry",
        "search engine web platform and online advertising",
        "digital privacy regulation and technology companies",
        "cloud computing data center and enterprise software",
        "electric vehicle battery and clean technology",
        "scientists discover new planet and astronomy news",
        "biotechnology drug research and clinical trial",
        "computer virus malware and security software",
        "technology startup product launch and innovation",
        "new gadget review and consumer electronics",
    ],
}

queries = []
for category, texts in QUERY_TEMPLATES.items():
    for text in texts:
        queries.append(
            {
                "query_id": f"q-{len(queries):04d}",
                "text": text,
                "filters": {
                    "category": category,
                    "doc_type": "article",
                    "year_gte": 2020,
                },
            }
        )

out_path = Path("data/queries.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    for q in queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
