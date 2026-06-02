package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass13 {
    private final String clinicId;
    private final List<String> doctors = new ArrayList<>();
    private final List<String> appointments = new ArrayList<>();
    private String leadDoctor;
    private int openSlots;
    private int emergencySlots;
    private boolean acceptingPatients;

    public LargeClass13(String clinicId, String leadDoctor) {
        this.clinicId = clinicId;
        this.leadDoctor = leadDoctor;
        this.acceptingPatients = true;
    }

    public void hireDoctor(String doctor) {
        doctors.add(doctor);
    }

    public void book(String patient) {
        if (acceptingPatients && openSlots > 0) {
            appointments.add(patient);
            openSlots--;
        }
    }

    public void reserveEmergencySlot() {
        emergencySlots++;
    }

    public void close() {
        acceptingPatients = false;
    }

    public String scheduleSummary() {
        return clinicId + ":" + leadDoctor + ":" + doctors.size() + ":" + appointments.size() + ":" + openSlots + ":" + emergencySlots + ":" + acceptingPatients;
    }
}
