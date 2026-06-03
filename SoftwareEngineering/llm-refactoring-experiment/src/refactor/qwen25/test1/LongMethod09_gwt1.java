package experiment.longmethod;

public class LongMethod09_gwt1 {
    public String formatAddress(String street, String city, String zip, boolean uppercase) {
        String value = concatenateAddress(street, city, zip);
        value = toUpperCaseIfRequired(value, uppercase);
        value = truncateIfLong(value);
        value = removeExtraSpaces(value);
        return value;
    }

    private String concatenateAddress(String street, String city, String zip) {
        return street + ", " + city + " " + zip;
    }

    private String toUpperCaseIfRequired(String value, boolean uppercase) {
        if (uppercase) {
            value = value.toUpperCase();
        }
        return value;
    }

    private String truncateIfLong(String value) {
        if (value.length() > 30) {
            value = value.substring(0, 30);
        }
        return value;
    }

    private String removeExtraSpaces(String value) {
        if (value.contains("  ")) {
            value = value.replace("  ", " ");
        }
        return value;
    }
}