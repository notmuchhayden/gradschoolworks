package experiment.longmethod;

public class LongMethod09 {
    public String formatAddress(String street, String city, String zip, boolean uppercase) {
        String value = street + ", " + city + " " + zip;
        if (uppercase) {
            value = value.toUpperCase();
        }
        if (value.length() > 30) {
            value = value.substring(0, 30);
        }
        if (value.contains("  ")) {
            value = value.replace("  ", " ");
        }
        return value;
    }
}
