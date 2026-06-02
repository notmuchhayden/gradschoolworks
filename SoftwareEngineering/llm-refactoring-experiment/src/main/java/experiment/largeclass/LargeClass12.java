package experiment.largeclass;

import java.util.ArrayList;
import java.util.List;

public class LargeClass12 {
    private final String recipeId;
    private String title;
    private final List<String> ingredients = new ArrayList<>();
    private final List<String> steps = new ArrayList<>();
    private int servings;
    private int calories;
    private boolean vegetarian;
    private boolean published;

    public LargeClass12(String recipeId, String title) {
        this.recipeId = recipeId;
        this.title = title;
    }

    public void addIngredient(String ingredient) {
        ingredients.add(ingredient);
    }

    public void addStep(String step) {
        steps.add(step);
    }

    public void scale(int servings) {
        this.servings = servings;
        calories = calories * servings;
    }

    public void markVegetarian(boolean vegetarian) {
        this.vegetarian = vegetarian;
    }

    public void publish() {
        published = true;
    }

    public String preview() {
        return recipeId + ":" + title + ":" + ingredients.size() + ":" + steps.size() + ":" + servings + ":" + calories + ":" + vegetarian + ":" + published;
    }
}
