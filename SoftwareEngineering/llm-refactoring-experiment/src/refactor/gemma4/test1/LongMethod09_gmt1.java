package experiment.longmethod;

public class LongMethod09_gmt1 {
    public String formatAddress(String street, String city, String zip, boolean uppercase) {
        String address = buildAddressString(street, city, zip);
        
        if (uppercase) {
            address = convertToUpperCase(address);
        }
        
        address = truncateAddress(address, 30);
        address = normalizeSpaces(address);
        
        return address;
    }

    private String buildAddressString(String street, String city, String zip) {
        return street + ", " + city + " " + zip;
    }

    private String convertToUpperCase(String text) {
        return text.toUpperCase();
    }

    private String truncateAddress(String text, int maxLength) {
        if (text.length() > maxLength) {
            return text.substring(0, maxLength);
        }
        return text;
    }

    private String normalizeSpaces(String text) {
        if (text.contains("  ")) {
            return text.replace("  ", " ");
        }
        return text;
    }
}