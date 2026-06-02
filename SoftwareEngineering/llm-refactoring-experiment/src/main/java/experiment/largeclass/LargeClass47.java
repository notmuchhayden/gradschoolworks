package experiment.largeclass;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class LargeClass47 {
    private final String payrollId;
    private final List<String> employees = new ArrayList<>();
    private final List<String> adjustments = new ArrayList<>();
    private final Map<String, Double> employeePay = new LinkedHashMap<>();
    private final Map<String, Integer> departmentHeadcount = new LinkedHashMap<>();
    private String accountant;
    private double grossPay;
    private double deductions;
    private double bonuses;
    private double employerTax;
    private int corrections;
    private boolean locked;

    public LargeClass47(String payrollId, String accountant) {
        this.payrollId = payrollId;
        this.accountant = accountant;
    }

    public void addEmployee(String employee) {
        employees.add(employee);
        employeePay.put(employee, 0.0);
    }

    public void addGrossPay(double amount) {
        grossPay += amount;
        adjustments.add("gross:" + amount);
    }

    public void addDeduction(double amount) {
        deductions += amount;
        adjustments.add("deduction:" + amount);
    }

    public void addBonus(double amount) {
        bonuses += amount;
        adjustments.add("bonus:" + amount);
    }

    public void assignDepartment(String department) {
        departmentHeadcount.merge(department, 1, Integer::sum);
    }

    public void applyEmployerTax(double amount) {
        employerTax += amount;
    }

    public void correctPayroll(String employee, double amount) {
        corrections++;
        employeePay.put(employee, employeePay.getOrDefault(employee, 0.0) + amount);
        adjustments.add("correction:" + employee + ":" + amount);
    }

    public void lock() {
        locked = true;
    }

    public String payrollState() {
        return payrollId + ":" + accountant + ":" + employees.size() + ":" + grossPay + ":" + deductions + ":" + bonuses + ":" + employerTax + ":" + corrections + ":" + locked;
    }

    public Map<String, Double> paySnapshot() {
        return new LinkedHashMap<>(employeePay);
    }
}
