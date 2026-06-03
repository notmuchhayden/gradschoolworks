package refactor.gemma4;

public class LongMethod07_gmt1 {
    public String classifyTemperature(double temp, double humidity, boolean cloudy) {
        String category = getTemperatureCategory(temp);
        category = appendHumidityStatus(category, humidity);
        category = appendCloudStatus(category, cloudy);
        return category;
    }

    // 1. 온도에 따른 기본 카테고리 결정
    private String getTemperatureCategory(double temp) {
        if (temp < 0) {
            return "freezing";
        } else if (temp < 10) {
            return "cold";
        } else if (temp < 25) {
            return "mild";
        } else {
            return "hot";
        }
    }

    // 2. 습도 상태에 따른 접미사 추가
    private String appendHumidityStatus(String category, double humidity) {
        if (humidity > 80) {
            return category + "-humid";
        } else if (humidity < 30) {
            return category + "-dry";
        }
        return category;
    }

    // 3. 구름 여부에 따른 접미사 추가
    private String appendCloudStatus(String category, boolean cloudy) {
        if (cloudy) {
            return category + "-cloudy";
        }
        return category;
    }
}