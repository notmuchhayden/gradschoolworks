package experiment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

class RefactoringBehaviorParityTest {
    private static final Path PROJECT_ROOT = Path.of("").toAbsolutePath();
    private static final List<String> MODELS = List.of("qwen25", "namotron3", "gemma4");
    private static final List<String> SAMPLE_NAMES = Stream.of("DuplicateCode", "LongMethod", "LargeClass")
            .flatMap(prefix -> Stream.iterate(1, i -> i + 1)
                    .limit(10)
                    .map(i -> prefix + String.format("%02d", i)))
            .toList();

    @TestFactory
    Stream<DynamicTest> refactoredOutputsMatchOriginalSamples() {
        return MODELS.stream()
                .flatMap(model -> SAMPLE_NAMES.stream()
                        .map(sample -> DynamicTest.dynamicTest(model + " " + sample,
                                () -> assertRefactoringMatches(model, sample))));
    }

    private static void assertRefactoringMatches(String model, String sample) throws Exception {
        try {
            Class<?> originalClass = Class.forName("experiment." + sample);
            Class<?> refactoredClass = compileAndLoad(model, sample);
            runScenario(sample, originalClass, refactoredClass);
        } catch (AssertionError e) {
            throw e;
        } catch (Exception e) {
            fail("Refactored API could not be exercised for " + model + " " + sample + ": " + e, e);
        }
    }

    private static Class<?> compileAndLoad(String model, String sample) throws IOException, ClassNotFoundException {
        Path source = PROJECT_ROOT.resolve("src/refactor")
                .resolve(model)
                .resolve(refactoredFileName(model, sample));
        Path output = Files.createTempDirectory("refactoring-parity-");
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        assertNotNull(compiler, "Tests must run on a JDK, not a JRE");

        ByteArrayOutputStream compilerOutput = new ByteArrayOutputStream();
        int result = compiler.run(
                null,
                compilerOutput,
                compilerOutput,
                "--release",
                "17",
                "-encoding",
                StandardCharsets.UTF_8.name(),
                "-classpath",
                System.getProperty("java.class.path"),
                "-d",
                output.toString(),
                source.toString());
        if (result != 0) {
            fail("Refactored source did not compile: " + source + System.lineSeparator()
                    + compilerOutput.toString(StandardCharsets.UTF_8));
        }

        URLClassLoader loader = new URLClassLoader(new URL[] {output.toUri().toURL()});
        return Class.forName(refactoredQualifiedName(model, sample), true, loader);
    }

    private static String refactoredFileName(String model, String sample) {
        return sample + suffix(model, sample) + ".java";
    }

    private static String refactoredQualifiedName(String model, String sample) {
        return "refactor." + model + "." + sample + suffix(model, sample);
    }

    private static String suffix(String model, String sample) {
        return switch (model) {
            case "qwen25" -> sample.startsWith("LongMethod") ? "_gwt1" : "_qwt1";
            case "namotron3" -> sample.startsWith("DuplicateCode") ? "_nmt1" : "_mnt1";
            case "gemma4" -> "_gmt1";
            default -> throw new IllegalArgumentException("Unknown model: " + model);
        };
    }

    private static void runScenario(String sample, Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        switch (sample) {
            case "DuplicateCode01" -> compareStaticMethods(originalClass, refactoredClass, types(int.class), values(-1, 0, 7), "score", "scoreAgain");
            case "DuplicateCode02" -> compareStaticMethods(originalClass, refactoredClass, types(int.class), values(-5, 0, 7), "clampAndScore", "clampAndScoreAgain");
            case "DuplicateCode03" -> compareStaticMethods(originalClass, refactoredClass, types(String.class), values("  alpha  ", "z"), "wrap", "wrapAgain");
            case "DuplicateCode04" -> compareStaticMethods(originalClass, refactoredClass, types(String.class), values("abc", ""), "mirror", "mirrorAgain");
            case "DuplicateCode05" -> compareStaticMethods(originalClass, refactoredClass, types(int[].class), values(new int[] {1, 2, 3}, new int[] {-2, 5}), "doubledSum", "doubledSumAgain");
            case "DuplicateCode06" -> compareStaticMethods(originalClass, refactoredClass, types(int[].class), values(new int[] {1, 9, 3}, new int[] {-4, -2}), "cappedMax", "cappedMaxAgain");
            case "DuplicateCode07" -> compareStaticMethods(originalClass, refactoredClass, types(List.class), values(List.of("a", "b", "c"), List.of()), "join", "joinAgain");
            case "DuplicateCode08" -> compareStaticMethods(originalClass, refactoredClass, types(List.class), values(List.of("a", " ", "b"), List.of("", "x")), "countNonEmpty", "countNonEmptyAgain");
            case "DuplicateCode09" -> compareStaticMethods(originalClass, refactoredClass, types(String.class, String.class), values(args("a", "b"), args("same", "same")), "tally", "tallyAgain");
            case "DuplicateCode10" -> {
                compareStatic(originalClass, refactoredClass, "report", types(List.class), values(List.of(1, 2, 3), List.of(-1, 4)));
                compareStatic(originalClass, refactoredClass, "reportAgain", types(), values(args()));
            }
            case "LongMethod01" -> compareInstance(originalClass, refactoredClass, "process", types(String.class, int.class), values(args(" xylophone ", 150), args("Zip", 10)));
            case "LongMethod02" -> compareInstance(originalClass, refactoredClass, "normalizeAndSum", types(int[].class), values(new int[] {-1, 4, 7, 60}, new int[] {}));
            case "LongMethod03" -> compareInstance(originalClass, refactoredClass, "buildStatus", types(String.class, boolean.class, int.class, double.class), values(args("svc", true, 0, 20.0), args("api", false, 4, 140.0)));
            case "LongMethod04" -> compareInstance(originalClass, refactoredClass, "estimatePrice", types(double.class, int.class, boolean.class, String.class), values(args(80.0, 4, true, "EU"), args(10.0, 2, false, "KR")));
            case "LongMethod05" -> compareInstance(originalClass, refactoredClass, "renderReport", types(String.class, int.class, int.class, int.class), values(args("scores", 40, 30, 20), args("low", 1, 2, 3)));
            case "LongMethod06" -> compareInstance(originalClass, refactoredClass, "checkAccess", types(String.class, int.class, boolean.class), values(args("admin", 23, true), args("editor", 10, true), args("viewer", 12, false), args("viewer", 24, false)));
            case "LongMethod07" -> compareInstance(originalClass, refactoredClass, "classifyTemperature", types(double.class, double.class, boolean.class), values(args(-5.0, 85.0, true), args(30.0, 20.0, false)));
            case "LongMethod08" -> compareInstance(originalClass, refactoredClass, "computeTax", types(int.class, int.class, boolean.class, String.class), values(args(1000, 0, true, "Seoul"), args(100, 4, false, "Daegu")));
            case "LongMethod09" -> compareInstance(originalClass, refactoredClass, "formatAddress", types(String.class, String.class, String.class, boolean.class), values(args("Long Street Name", "Seoul", "12345", true), args("A", "B", " 9", false)));
            case "LongMethod10" -> compareInstance(originalClass, refactoredClass, "convertAndRound", types(double.class, String.class), values(args(6.2, "km"), args(155.4, "cm"), args(2.6, "m")));
            case "LargeClass01" -> compareLargeClass01(originalClass, refactoredClass);
            case "LargeClass02" -> compareLargeClass02(originalClass, refactoredClass);
            case "LargeClass03" -> compareLargeClass03(originalClass, refactoredClass);
            case "LargeClass04" -> compareLargeClass04(originalClass, refactoredClass);
            case "LargeClass05" -> compareLargeClass05(originalClass, refactoredClass);
            case "LargeClass06" -> compareLargeClass06(originalClass, refactoredClass);
            case "LargeClass07" -> compareLargeClass07(originalClass, refactoredClass);
            case "LargeClass08" -> compareLargeClass08(originalClass, refactoredClass);
            case "LargeClass09" -> compareLargeClass09(originalClass, refactoredClass);
            case "LargeClass10" -> compareLargeClass10(originalClass, refactoredClass);
            default -> throw new IllegalArgumentException("No scenario for " + sample);
        }
    }

    private static void compareStatic(Class<?> originalClass, Class<?> refactoredClass, String methodName, Class<?>[] parameterTypes, Object[] cases) throws Exception {
        Method original = originalClass.getMethod(methodName, parameterTypes);
        Method refactored = refactoredClass.getMethod(methodName, parameterTypes);
        for (Object testCase : cases) {
            Object[] args = normalizeArgs(parameterTypes, testCase);
            assertSameValue(methodName, invoke(original, null, args), invoke(refactored, null, args));
        }
    }

    private static void compareStaticMethods(Class<?> originalClass, Class<?> refactoredClass, Class<?>[] parameterTypes, Object[] cases, String... methodNames) throws Exception {
        for (String methodName : methodNames) {
            compareStatic(originalClass, refactoredClass, methodName, parameterTypes, cases);
        }
    }

    private static void compareInstance(Class<?> originalClass, Class<?> refactoredClass, String methodName, Class<?>[] parameterTypes, Object[] cases) throws Exception {
        Object originalTarget = newInstance(originalClass);
        Object refactoredTarget = newInstance(refactoredClass);
        Method original = originalClass.getMethod(methodName, parameterTypes);
        Method refactored = refactoredClass.getMethod(methodName, parameterTypes);
        for (Object testCase : cases) {
            Object[] args = normalizeArgs(parameterTypes, testCase);
            assertSameValue(methodName, invoke(original, originalTarget, args), invoke(refactored, refactoredTarget, args));
        }
    }

    private static void compareLargeClass01(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class), "C1", "Kim");
        Object refactored = newInstance(refactoredClass, types(String.class, String.class), "C1", "Kim");
        callBoth(originalClass, refactoredClass, original, refactored, "deposit", types(double.class), 120.0);
        callBoth(originalClass, refactoredClass, original, refactored, "withdraw", types(double.class), 20.0);
        callBoth(originalClass, refactoredClass, original, refactored, "rename", types(String.class), "Lee");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "tier", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "summary", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "notes", types());
        callBoth(originalClass, refactoredClass, original, refactored, "deactivate", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "summary", types());
    }

    private static void compareLargeClass02(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class), "T1", "Kim");
        Object refactored = newInstance(refactoredClass, types(String.class, String.class), "T1", "Kim");
        callBoth(originalClass, refactoredClass, original, refactored, "escalate", types());
        callBoth(originalClass, refactoredClass, original, refactored, "assign", types(String.class), "Lee");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "isHighPriority", types());
        assertEquals(reportWithoutTimestamp(call(originalClass, original, "report", types())),
                reportWithoutTimestamp(call(refactoredClass, refactored, "report", types())));
        assertCallEquals(originalClass, refactoredClass, original, refactored, "transitionStats", types());
        callBoth(originalClass, refactoredClass, original, refactored, "close", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "transitionStats", types());
    }

    private static List<String> reportWithoutTimestamp(Object report) {
        List<String> parts = Arrays.asList(String.valueOf(report).split("\\|"));
        return parts.subList(0, Math.min(4, parts.size()));
    }

    private static void compareLargeClass03(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class), "R1", "Kim");
        Object refactored = newInstance(refactoredClass, types(String.class, String.class), "R1", "Kim");
        callBoth(originalClass, refactoredClass, original, refactored, "addStop", types(String.class), "A");
        callBoth(originalClass, refactoredClass, original, refactored, "reroute", types(String.class), "B");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "routeSummary", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "averageKmPerStop", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "stops", types());
        callBoth(originalClass, refactoredClass, original, refactored, "lock", types());
        callBoth(originalClass, refactoredClass, original, refactored, "addStop", types(String.class), "C");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "routeSummary", types());
    }

    private static void compareLargeClass04(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class), "SE101", "Kim");
        Object refactored = newInstance(refactoredClass, types(String.class, String.class), "SE101", "Kim");
        callBoth(originalClass, refactoredClass, original, refactored, "addQuizScore", types(int.class), 90);
        callBoth(originalClass, refactoredClass, original, refactored, "addQuizScore", types(int.class), 70);
        callBoth(originalClass, refactoredClass, original, refactored, "submitHomework", types());
        callBoth(originalClass, refactoredClass, original, refactored, "publish", types(String.class), "B101");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "averageScore", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "status", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "isStrict", types());
    }

    private static void compareLargeClass05(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class), "A1", "Kim");
        Object refactored = newInstance(refactoredClass, types(String.class, String.class), "A1", "Kim");
        callBoth(originalClass, refactoredClass, original, refactored, "addBudget", types(double.class), 300.0);
        callBoth(originalClass, refactoredClass, original, refactored, "allocate", types(double.class), 100.0);
        callBoth(originalClass, refactoredClass, original, refactored, "consume", types(double.class), 40.0);
        assertCallEquals(originalClass, refactoredClass, original, refactored, "snapshot", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "remaining", types());
        callBoth(originalClass, refactoredClass, original, refactored, "freeze", types());
        callBoth(originalClass, refactoredClass, original, refactored, "allocate", types(double.class), 50.0);
        assertCallEquals(originalClass, refactoredClass, original, refactored, "snapshot", types());
    }

    private static void compareLargeClass06(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class), "L1", "Kim");
        Object refactored = newInstance(refactoredClass, types(String.class, String.class), "L1", "Kim");
        callBoth(originalClass, refactoredClass, original, refactored, "addBook", types(String.class), "BookA");
        callBoth(originalClass, refactoredClass, original, refactored, "addBook", types(String.class), "BookB");
        callBoth(originalClass, refactoredClass, original, refactored, "borrow", types(String.class, String.class), "Lee", "BookA");
        callBoth(originalClass, refactoredClass, original, refactored, "markLost", types());
        callBoth(originalClass, refactoredClass, original, refactored, "markDamaged", types());
        callBoth(originalClass, refactoredClass, original, refactored, "renameLibrarian", types(String.class), "Park");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "inventorySummary", types());
    }

    private static void compareLargeClass07(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class), "S1", "solo");
        Object refactored = newInstance(refactoredClass, types(String.class, String.class), "S1", "solo");
        callBoth(originalClass, refactoredClass, original, refactored, "addPlayer", types(String.class), "Kim");
        callBoth(originalClass, refactoredClass, original, refactored, "submitScore", types(int.class), 50);
        callBoth(originalClass, refactoredClass, original, refactored, "submitScore", types(int.class), 70);
        assertCallEquals(originalClass, refactoredClass, original, refactored, "dashboard", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "averageScore", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "players", types());
        callBoth(originalClass, refactoredClass, original, refactored, "finish", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "dashboard", types());
    }

    private static void compareLargeClass08(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class), "SN1");
        Object refactored = newInstance(refactoredClass, types(String.class), "SN1");
        callBoth(originalClass, refactoredClass, original, refactored, "calibrate", types(double.class), 10.0);
        callBoth(originalClass, refactoredClass, original, refactored, "record", types(double.class), 9.0);
        callBoth(originalClass, refactoredClass, original, refactored, "record", types(double.class), 25.0);
        assertCallEquals(originalClass, refactoredClass, original, refactored, "movingAverage", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "status", types());
    }

    private static void compareLargeClass09(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class), "P1", "Kim");
        Object refactored = newInstance(refactoredClass, types(String.class, String.class), "P1", "Kim");
        callBoth(originalClass, refactoredClass, original, refactored, "addTrack", types(String.class), "A");
        callBoth(originalClass, refactoredClass, original, refactored, "addTrack", types(String.class), "B");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "nextTrack", types());
        callBoth(originalClass, refactoredClass, original, refactored, "skip", types());
        callBoth(originalClass, refactoredClass, original, refactored, "repeat", types());
        callBoth(originalClass, refactoredClass, original, refactored, "shuffle", types(boolean.class), true);
        assertCallEquals(originalClass, refactoredClass, original, refactored, "overview", types());
    }

    private static void compareLargeClass10(Class<?> originalClass, Class<?> refactoredClass) throws Exception {
        Object original = newInstance(originalClass, types(String.class, String.class, int.class), "W1", "Kim", 2);
        Object refactored = newInstance(refactoredClass, types(String.class, String.class, int.class), "W1", "Kim", 2);
        callBoth(originalClass, refactoredClass, original, refactored, "store", types(String.class), "A");
        callBoth(originalClass, refactoredClass, original, refactored, "store", types(String.class), "B");
        callBoth(originalClass, refactoredClass, original, refactored, "store", types(String.class), "C");
        callBoth(originalClass, refactoredClass, original, refactored, "dispatch", types(String.class), "A");
        callBoth(originalClass, refactoredClass, original, refactored, "relabel", types(String.class), "Lee");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "report", types());
        assertCallEquals(originalClass, refactoredClass, original, refactored, "shipments", types());
        callBoth(originalClass, refactoredClass, original, refactored, "lock", types());
        callBoth(originalClass, refactoredClass, original, refactored, "dispatch", types(String.class), "B");
        assertCallEquals(originalClass, refactoredClass, original, refactored, "report", types());
    }

    private static void callBoth(Class<?> originalClass, Class<?> refactoredClass, Object original, Object refactored, String methodName, Class<?>[] parameterTypes, Object... args) throws Exception {
        call(originalClass, original, methodName, parameterTypes, args);
        call(refactoredClass, refactored, methodName, parameterTypes, args);
    }

    private static void assertCallEquals(Class<?> originalClass, Class<?> refactoredClass, Object original, Object refactored, String methodName, Class<?>[] parameterTypes) throws Exception {
        assertSameValue(methodName,
                call(originalClass, original, methodName, parameterTypes),
                call(refactoredClass, refactored, methodName, parameterTypes));
    }

    private static Object call(Class<?> type, Object target, String methodName, Class<?>[] parameterTypes, Object... args) throws Exception {
        Method method = type.getMethod(methodName, parameterTypes);
        return invoke(method, target, args);
    }

    private static Object invoke(Method method, Object target, Object... args) throws Exception {
        try {
            return method.invoke(target, args);
        } catch (InvocationTargetException e) {
            throw rethrowCause(e);
        }
    }

    private static Exception rethrowCause(InvocationTargetException e) throws Exception {
        Throwable cause = e.getCause();
        if (cause instanceof Exception exception) {
            throw exception;
        }
        if (cause instanceof Error error) {
            throw error;
        }
        throw new RuntimeException(cause);
    }

    private static Object newInstance(Class<?> type) throws Exception {
        return newInstance(type, types());
    }

    private static Object newInstance(Class<?> type, Class<?>[] parameterTypes, Object... args) throws Exception {
        Constructor<?> constructor = type.getConstructor(parameterTypes);
        try {
            return constructor.newInstance(args);
        } catch (InvocationTargetException e) {
            throw rethrowCause(e);
        }
    }

    private static Class<?>[] types(Class<?>... types) {
        return types;
    }

    private static Object[] values(Object... values) {
        return values;
    }

    private static Object[] args(Object... args) {
        return args;
    }

    private static Object[] normalizeArgs(Class<?>[] parameterTypes, Object testCase) {
        if (parameterTypes.length == 0) {
            return args();
        }
        if (parameterTypes.length == 1) {
            return new Object[] {testCase};
        }
        return (Object[]) testCase;
    }

    private static void assertSameValue(String label, Object expected, Object actual) {
        if (expected instanceof Double expectedDouble && actual instanceof Double actualDouble) {
            assertEquals(expectedDouble, actualDouble, 1.0e-9, label);
            return;
        }
        if (expected instanceof Float expectedFloat && actual instanceof Float actualFloat) {
            assertEquals(expectedFloat, actualFloat, 1.0e-6f, label);
            return;
        }
        assertEquals(expected, actual, label);
    }
}
