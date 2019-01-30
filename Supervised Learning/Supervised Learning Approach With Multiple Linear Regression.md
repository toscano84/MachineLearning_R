---
title: "Predicting Airfares on New Routes a Supervised Learning Approach With Multiple Linear Regression"
author: Hugo Toscano
date: '2018-10-24'
slug: Predicting Airfares on New Routes
categories: [Predictive Analytics]
tags: [machine learning, multiple linear regression, predictive analytics]
---

This post  will talk about multiple linear regression in the context of machine learning. Linear regression is one of the simplest and most used approaches for supervised learning. This tutorial will try to help you in how to use the linear regression algorithm. I am also new to the machine learning approach, but I'm very interested in this area given the predictive ability that you can gain from this. Let's hope I can help you. During the tutorial, we will build various multiple linear regression models. Next, we will evaluate these models,  choose the more accurate one and evaluate how well the model generalizes  to new data.

Throughout the post we will be using  data  extracted from the excellent book 'Data Mining for Business Analytics: Concepts, Techniques, and Applications in R'. The data is called Airfares and it's from the 90s. The data frame has several variables as you will be able to ascertain soon. The challenge related to this data frame is how the deregulation of airlines in the 70's, and consequently the entrance of new carriers with lower prices like Southwest airlines, influence the prices of airfares. The business problem is to predict how much it will cost the airfares in a new flight route. So, imagine that you are working for an airline company  that intends to open a new route in the USA and it has data related to the characteristics of airports, economic and demographic information about the cities where flights arrive and depart, flight distances and average price of air flights. The company to whom we work wants to open a new route, but doesn't know which average price should ask for. The goal of your job in this to find  a model that can better predict the average price of airfares.

## Loading and Exploring our Data Frame

Now, let's first load the packages to be used in our analysis.

```{r message=FALSE, warning=FALSE}
library(here) # set directory
library(tidyverse) # data wrangling and visualization
library(caret) # machine learning techniques
library(corrr) # visualize correlations
library(ggcorrplot) # visualize correlations
library(lmtest) # check homoscedasticity
library(MASS) # create stepwise regression models
library(car) # analyse homoscedasticity
library(yardstick) # check regression models metrics
library(broom) # tidy statistical models
options(scipen = 999) # disable scientific notation
```

Next , we open our dataset and start to explore it.

```{r message=FALSE, warning=FALSE}
# open file
airfares <- read_csv(here("airfares.csv")) %>% 
  rename_all(str_to_lower) # all variables' names to lower case

# explore structure of the data frame
glimpse(airfares)
```

A `glimpse()` of our variables show that we have 18 variables. 
In the following table copied from the book 'Data Mining for Business Analytics', you can see a more extensive description of each variable.

![](/post/2018-10-23-predicting-airfares-prices-on-new-routes-a-supervised-learning-approach-with-multiple-linear-regression_files/variables.PNG){width=80%}


Let's further explore our dataset by using the `summary()` function to check our data. 
```{r}
# explore dataset
summary(airfares)

```

Moreover, we will check if there are any missing values.
```{r}
sum(is.na(airfares)) # check if there are missing values
```

We don't have any missing values.

Now we can  visualize the distribution of our dependent variable named fare.

```{r message=FALSE, warning=FALSE}
# visualize our dependent variable
ggplot(airfares, aes(fare)) +
  geom_histogram(binwidth = 10)
```


Next, we will remove the first four variables from our data frame (s_code, s_city, e_code, and e_city).

```{r}
# remove variables 
airfares <- airfares %>%
  dplyr::select(-c(1:4)) # remove variables 1 to 4. These correspond to variables (s_code, s_city, e_code, and e_city)


glimpse(airfares)
```

Now, we will keep exploring the data and check for correlations between the numeric variables of our data frame.

We will use the package `ggcorrplot` for this.

```{r}
# check for correlations

# create correlations dataframe
cor_df <- airfares %>% 
  select_if(is.numeric) %>%
  cor()

# visualize correlations with ggcorrplot
ggcorrplot(cor_df, hc.order = TRUE, type = "lower", 
           lab = TRUE)

```

From these correlations, we can verify that the prices of airlines have a high positive correlation with the variables coupon and distance. Note: coupon and distance are highly correlated which can be a problem due to the multicollinearity assumption of the regression. 

Here with the function `network_plot()` from the `corrr` package we can see that above 0.3, only distance, coupon and e_income are correlated with the variable fare.

```{r}
# network plot
cor_df %>%
  network_plot(min_cor = 0.3) # check network plot of correlations above 0.3 in the positive and negative direction

```

## Creation of Regression Models

In the following step, **we will create a train and a test dataset**, so that we can fit our training model and apply it to a test dataset in order to validate it and see whether it generalizes into a new data frame.
We will use the `createDataPartition()` function from the `caret` package to make this partition.
```{r}
# partition the data
set.seed(1234)
partition <- createDataPartition(airfares$fare, p = 0.7, list = FALSE) # 70% corresponds to the train data frame and 30% to the data frame

# create train and test dataframes
train_airfares <- airfares[partition, ]

test_airfares <- airfares[-partition, ]

```


Let us now create our first regression model with our training set. We will name it **model0**.

```{r}
# creation of model0
model0 <- lm(fare ~ ., data = train_airfares)
summary(model0)
```



We can see that all variables, but coupon and new, significantly predict the mean price of airlines flights. As mentioned before, this data has in mind the entrance of new players with low prices, as the Southwest Airlines, in the price of tickets flights. The regression shows that the presence of Southwest Airlines(SW) can decrease the mean price of fares in 39.338$.

As an example , we can check some assumptions of the regression in our model. Note: We will not do it  for the models created afterwards. It's just for you to have an idea on how to test some of the regression assumptions.
First, we can check for multicollinearity, that is, if there is a high linear association between the predictor variables of our model. To check it , we can use the `vif()` function.


```{r}
# test assumption of multicollinearity
car::vif(model0)
```

All values are below 5,  meaning that we don't have multicollinearity in our model.

Second, we can check if our residuals show **homoscedasticity**. We will use the `augment()` function from the `broom` package to get the predicted and residuals of our model. After that we will visualize how the residuals distribute themselves.

```{r}
# test assumptions of heteroskedasticity
aug <- augment(model0)

# visualiza the residuals
aug %>%
  ggplot(aes(x = .fitted, y = .resid)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  ggtitle("Residuals vs Fitted")
```


It looks like  we have heteroscedasticity. The residuals are not distributed uniformly along zero. Using the `bptest()` function from the `lmtest` package we  can come to that conclusion.

```{r}
lmtest::bptest(model0)
```


The p-value is below 0.05, so the assumption of homoscedasticity is not satisfied. This result may suggest an incomplete model, but for the purposes of this post we will not give it too much attention. Nonetheless, you should know how to check the regression assumptions.

Putting aside the assumptions of the regression, let's go back to fit our training data. Now, we will use a stepwise regression method, in order to find a model that minimizes the error. We can use the `stepAIC()` function from the `MASS` package.


```{r}
# create stepwise regression model
step_model <- stepAIC(model0, direction = "both")
```


Now we can check which variables were dropped in the stepwise regression:
```{r}
step_model
```

This function reflects a model that dropped the variables coupon and new.
Let's check this model and call it **model1**.

```{r}
model1 <- lm(fare ~ vacation + sw + hi + s_income + e_income + s_pop + e_pop + slot + gate + distance + pax, data = train_airfares)

summary(model1)
```


All variables significantly predict the airfares.

So, we have 2 models - **model0** and **model1** - and now we will create two more models - **model2** and **model3**. Afterwards, we will assess the performance metrics of each one of the 4 models.

For  **model2**, we will remove the variables s_pop, e_pop, slot, and gate.

```{r}
# create model2
model2 <- lm(fare ~ vacation + sw + hi + s_income + e_income + distance + pax, data = train_airfares)

summary(model2)
```


For  **model3**, we will keep only the variables vacation, sw, distance, and pax.

```{r}
# create model3
model3 <- lm(fare ~ vacation + sw + distance + pax, data = train_airfares)

summary(model3)
```

## Assssment of the Regression Models
So, we've built our models, however we haven't assessed which one performs better. In the following assessment, we will center our attention in two key performance metrics: the **root mean squared error(RMSE)** and the **R-squared**. I will show you how to assess the models with the `caret` and `yardstick` packages. 

```{r}
# Assessing model Performance
# model0
lm0 <- train(fare ~ ., method = "lm", data = train_airfares,
                trControl = trainControl(method = "none"))

# model1
lm1 <- train(fare ~ vacation + sw + hi + s_income + e_income + 
               s_pop + e_pop + slot + gate + distance + pax, method = "lm", data = train_airfares,
                 trControl = trainControl(method = "none"))

# model2
lm2 <- train(fare ~ vacation + sw + hi + s_income + e_income + distance + pax, method = "lm", data = train_airfares,
                 trControl = trainControl(method = "none"))

# model3
lm3 <- train(fare ~ vacation + sw + distance + pax, method = "lm", data = train_airfares,
                 trControl = trainControl(method = "none"))


# create dataframe with the 4 models
agg_data <- train_airfares %>%
  mutate(regression0 = predict(lm0, train_airfares),
         regression1 = predict(lm1, train_airfares),
         regression2 = predict(lm2, train_airfares),
         regression3 = predict(lm3, train_airfares))

# use the function metrics from the yardstick package to assess the models
# metrics model0
metrics(agg_data, truth = fare, estimate = regression0)
# metrics model1
metrics(agg_data, truth = fare, estimate = regression1)
# metrics model2
metrics(agg_data, truth = fare, estimate = regression2)
# metrics model3
metrics(agg_data, truth = fare, estimate = regression3)

```

Thus, the metrics of these models show that  **model0** and **model1** are the best ones. They have a lower **RMSE** and a higher **R-squared**. Between these two, we should choose model1 as it  includes less variables. Nonetheless, we are still left with 2 steps.

First step, let's check these 4 models visually.

```{r}
# visualized assessment of the models
agg_data %>%
  gather(type_of_lm, output, regression0:regression3) %>%
  ggplot(aes(fare, output, color = type_of_lm)) +
  geom_point(size = 1.5, alpha = 0.5) +
  facet_wrap(~type_of_lm) +
  geom_abline(lty = 2, color = "gray50") +
  geom_smooth(method = "lm")
```

It supports what we have said before, **models 0 and 1** are the best ones. The regression line of both models is closer to the grey dashed line than the models 2 and 3.

## Generalization of the Regression Models to new Data

The last step is a critical one in machine learning. 

Firstly, we have created and fitted the models. 

Secondly, we have assessed the models and it seems **model0** and **model1** are the best fit. However , we still do not know how the models compare when making predictions on new data. So, in this last step we need to check how our models work in our test data.

```{r}
# generalization to new data
# create dataframe with the 4 models
tests_agg_data <- test_airfares %>%
  mutate(regression0t = predict(lm0, test_airfares),
         regression1t = predict(lm1, test_airfares),
         regression2t = predict(lm2, test_airfares),
         regression3t = predict(lm3, test_airfares))

# use the yardstick package to check metrics
# metrics model0
metrics(tests_agg_data, truth = fare, estimate = regression0t)
# metrics model1
metrics(tests_agg_data, truth = fare, estimate = regression1t)
# metrics model2
metrics(tests_agg_data, truth = fare, estimate = regression2t)
# metrics model3
metrics(tests_agg_data, truth = fare, estimate = regression3t)

```


We should also visualize our models.

```{r}
# visualized assessment of the models
tests_agg_data %>%
  gather(type_of_lm, output, regression0t:regression3t) %>%
  ggplot(aes(fare, output, color = type_of_lm)) +
  geom_point(size = 1.5, alpha = 0.5) +
  facet_wrap(~type_of_lm) +
  geom_abline(lty = 2, color = "gray50") +
  geom_smooth(method = "lm")
```


The generalization shows that **models0** and **models1** still have a higher **R-squared** and a lower **RMSE** that the other models. However, we can attest that in test data the **RMSE** is higher and the **R-squared** is lower than in the train data. This means our model is slightly overfit.

## Choosing the Model
In sum, we have created, assessed and evaluated our models. Now, we need to choose the model. Given the lower RMSE and higher R-squared compared to **model2** and **model3**, as well as a lower number of variables than **model0**, we should choose **model1** to make predictions.

Now that we have our final model, **model1**, let us imagine that we needed to predict the average fare on a route with these characteristics: **vacation** = no, **sw** = yes, **hi** = 3980, **s_income** = $35,000, **e_income** = $45,344, **s_pop** = 3,975,003, **e_pop** = 6,327,987, **slot** = free, **gate** = free, **distance** = 2410 miles, **pax** = 14200.

First, we should create a new data frame with this data.

```{r message=FALSE, warning=FALSE}
# making predictions
new_df <- data.frame(vacation = "Yes", 
                     sw = "Yes",
                     hi = 3980,
                     s_income = 35000,
                     e_income = 45344,
                     s_pop = 3975003,
                     e_pop = 6327987,
                     slot = "Free",
                     gate = "Free",
                     distance = 2410,
                     pax = 14200)
```

Next, we should use the `predict()` function with two arguments: **model1** and the new data frame.

```{r}
predict(model1, newdata = new_df)
```

Our model predicts that a route with these characteristics will have an average price of **$245.169**.

That's the amazing thing with machine learning:  Its predictive ability constantly persuades me to learn more and more about it. Of course we should not take it at face value. Our model has limitations and a more advanced algorithm would likely given us more predictive power.

I hope you have enjoyed this post. I'm still giving my first steps in machine learning and in case  I have made any mistake please free to point it out. Thanks a lot and keep coding!



