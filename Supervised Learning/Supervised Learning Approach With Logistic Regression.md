---
title: "Creating a Model to Predict if a Bank Customer accepts Personal Loans"
author: Hugo Toscano
date: '2018-11-30'
slug: creating-a-model-to-increase-the-personal-loans-in-banks
categories: [Predictive Analytics]
tags: [machine learning, logistic regression, predictive analytics]
---

In this post, we will fit a multiple logistic regression model to predict the probability of a bank customer accepting a personal loan based on multiple variables to be described later. Logistic regression is a supervised learning algorithm were the independent variable has a qualitative nature. In this case, corresponding to the acceptance or rejection of a personal loan. This tutorial will build multiple logistic regression models and assess them.

The data called UniversalBank comes again from the handbook 'Data Mining for Business Analytics: Concepts, Techniques, and Applications in R'. The bank's business goal is to find the best combination of variables that can increase the probability of loan acceptance. 


## Data Exploration

First, we must load our libraries.

```{r message=FALSE, warning=FALSE}
library(here)
library(tidyverse) # data wrangling, data visualization
library(broom) # tidy statistics
library(caret) # apply machine learning algorithms
library(janitor) # tidy dataframes
library(MASS) # in this case it's used for the stepwise regression
library(readxl) # open excel files

options(scipen = 999) # number formatting to not include scientific notation
```

Afterwards, we load our data frame and explore it. We also need to transform into factors  a couple of our variables so that they are interpretable  by our logistic regression models.

```{r}
# open dataset
bank <- read_excel(here("UniversalBank.xlsx"), 
                   skip = 3,
                   sheet = 2) %>%
  clean_names() %>%
  mutate_at(vars(education, personal_loan, securities_account, cd_account,
                 online, credit_card), funs(as.factor))

glimpse(bank)
```

Here we can see in more detail each variable description:


![](/post/2018-10-23-predicting-airfares-prices-on-new-routes-a-supervised-learning-approach-with-multiple-linear-regression_files/variables_bank.PNG){width=80%}


Let's continue our data exploration. We can summarize our data:

```{r}
# explore dataset
summary(bank)


```

It's also important to check for any missing values.

```{r}
# check missing values
sum(is.na(bank))

```

Now that we have confirmed no values are missing, we can start with the visual exploration:

```{r}
# boxplot

bank %>%
  ggplot(aes(x = personal_loan, y = age, colour = personal_loan)) +
  geom_boxplot()

# histogram
bank %>%
  ggplot(aes(income, fill = personal_loan)) +
  geom_histogram(colour = "black")

```

It looks like the median age of persons who accepted the loan is similar to the median age of persons who did not accept it. However, the histogram shows that much less customers accepted the loan compared to the ones who rejected it. 

This class imbalance, that is, many more people whose loan was rejected than accepted can be an issue to be taken into account when building our model. This post will focus on it later on .

For now, we can compute this imbalance in a more concrete way.

```{r}
bank %>%
  count(personal_loan) %>%
  mutate(prop = n / sum(n))
```

Thus, only around 10% of our customers accepted the loan.


## Building Logistic Regression Models

In this section, **we will build our train and test datasets**. The goal is to fit our training model, apply it in a testing data frame for validation and to see if it generalizes to new data. In order to create a train and test dataset, we can use the `createDataPartition()` function from the `caret` package to make this partition. But before that we should tidy our data frame.

```{r}
bank_tidy <- bank %>%
  dplyr::select(-id, -zip_code) # delete columns id and zipcode - not relevant for the logistic regression model

glimpse(bank) # overview of the dataframe

bank_tidy$personal_loan <- factor(bank_tidy$personal_loan, levels=c(0,1), labels=c("No","Yes"))  # label our dependent variable 
```


At this time we create a train and test datasets:

```{r}
set.seed(1234)
partition <- createDataPartition(bank_tidy$personal_loan, 
                                 p = 0.7, 
                                 list = FALSE)

train <- bank_tidy[partition, ]
test <- bank_tidy[-partition, ]
```


Now, we will create our first logistic regression model called **model0**:

```{r}
model0 <- glm(personal_loan ~., data = train,
              family = "binomial")

summary(model0)
```

The model shows that all variables predict a bank costumer loan acceptance, but age, professional experience, and mortgage value. The coefficient estimates show a relationship between the predictors and the dependent variable on a log-odds scale.  For instance, an increase of one person in a family is associated with an increase in the log odds of a personal loan acceptance by 0.654 units. 

This can be interpreted in a more concrete manner by computing the odds.


```{r}
exp(coef(model0))

```

This shows that an increase of one family member, increases the odds of a bank costumer accepting a personal loan by a factor of 1.922.

Let's now check if we have outliers in our data set. In case we find them , we will not delete them; it's just something to have in mind while doing this type of analysis. **Note: we will do this analysis only for model0.** First, we should use the `augment()` function from the `broom` package to get the residuals data of our statistical model. In the next step, we should use `ggplot2` to visualize possible outliers.




```{r}
#- check for some assumptions
# Residual Assessment
tidy_mod0 <- broom::augment(model0) %>%
  mutate(index = 1:n())

tidy_mod0 %>%
  ggplot(aes(x = index, y = .std.resid, colour = personal_loan)) +
  geom_point(alpha = 0.4)

```

It looks like we have some cases above 3. Let's check how many there are and the corresponding number of the case.

```{r}
tidy_mod0 %>%
  filter(abs(.std.resid) > 3) %>%
  count()

plot(model0, which = 4, id.n = 8)
```

In total, we have 8 cases exceeding 3 and in the graph above we can see the number of the corresponding case.


Now, we should move on and focus in fitting our training data. For the creation of model1, we will use a stepwise regression with the function `stepAIC()` from the `MASS` package.


```{r message=FALSE, warning=FALSE}
# model1
step_log <- stepAIC(model0, direction = "both")


```

We can see which variables were kept for our model.

```{r}
step_log

```

Now, it's time to create our model1.

```{r}
model1 <- glm(personal_loan ~ income + family + cc_avg + education + 
                securities_account + cd_account + online + credit_card, family = "binomial", 
              data = train)

summary(model1)
```

All variables predict the loan acceptance. We will create one last model, called model2. The most important predictors present in model1 will be applied to this last model. To get the information about the most relevant predictors, we will use the `varImp()` from the `caret` package.

```{r}
caret::varImp(model1) %>%
  tibble::rownames_to_column("variable") %>%
  arrange(desc(Overall))
```

We will choose some of the most important variables and create model2.

```{r}
# model2
model2 <- glm(personal_loan ~ income + family + education + cd_account + credit_card + cc_avg + 
                online, family = "binomial", 
              data = train)

summary(model2)
```

## Assssment of the Logistic Regression Models

Now with the help of the `caret` package we have to assess which model has a better performance. We will use three key performance metrics: **accuracy**, **ppv**, or positive predicted values, and  **npv**, negative predicted values. Accuracy corresponds to the True Positives (TP) + the True Negatives(TN) divided by the TP + TN + False Positive(FP) + False Negatives(FN). PPV corresponds to the cases rightfully identified as positive(TP) divided by the TP + FP. The NPV is the number of cases rightfully identified as negative(TN) divided by the TN + FN.

It's now time to build our models  by using the function `confusionMatrix` to compute their metrics. **Note**: For each model, a sampling method called upsampling will be used due to the imbalance present in our dependent variable.

* **model0**



```{r}
# model0
glm0 <- train(personal_loan ~., method = "glm",
              family = "binomial",
              data = train,
              trControl = trainControl(method = "none", 
                                       sampling = "up")) # upsampling use because 

confusionMatrix(predict(glm0, 
                        train), train$personal_loan, positive = "Yes")


```


* **model1**

```{r}
glm1 <- train(personal_loan ~ income + family + cc_avg + education + 
    securities_account + cd_account + online + credit_card, method = "glm", 
              family = "binomial",
              data = train,
              trControl = trainControl(method = "none", 
                                       sampling = "up"))

confusionMatrix(predict(glm1, 
                        train), train$personal_loan, positive = "Yes")
```

* **model2**

```{r}
glm2 <- train(personal_loan ~ income + family + education + cd_account + 
    credit_card + cc_avg + online, method = "glm", 
              family = "binomial",
              data = train,
              trControl = trainControl(method = "none", 
                                       sampling = "up"))

confusionMatrix(predict(glm2, 
                        train), train$personal_loan, positive = "Yes")
```

Looking at the 3 confusion matrices, it seems that model2 has a higher accuracy, ppv, and npv. Nonetheless, model0 and model1 are also highly accurate. However, we still have to check how these models generalize to new data.

## Generalization of the Logistic Regression Models to new Data

Now, we should check how the models generalize to new data.

* **model0**

```{r}
# model0
confusionMatrix(predict(glm0, 
                        test), test$personal_loan, positive = "Yes")
```

* **model1**

```{r}
# model1
confusionMatrix(predict(glm1, 
                        test), test$personal_loan, positive = "Yes")

```

* **model2**
```{r warning=FALSE}
# model2
confusionMatrix(predict(glm2, 
                        test), test$personal_loan, positive = "Yes")
```

All models maintain or slightly increase their metrics in the test dataset. Thus, the metrics of all models are good.

Nonetheless, to keep our model simpler, we will choose model2. Now that we have our final model, it's time for some predictions. Let us imagine that we have two  bank customers. They have the same characteristics in relation to the variables of our model2, but one, income. While customer A has an annual income of 100 thousand dollars, customer B has a 45 thousand dollars annual income. We can use the `predict()` function to compare the probability of accepting a loan based on this income difference.



```{r}
predict(model2, data.frame(income = c(100, 45),
        family = c(3, 3),
        cc_avg = c(0.8, 0.8),
        education = c("3", "3"),
        securities_account = c("1", "1"),
        cd_account = c("1", "1"),
        credit_card = c("1", "1"),
        online = c("1", "1")), 
type = "response")
```

As a result, the probability of accepting a loan is of 56.67% for customer A, while for customer B is only 5.30%.

As mentioned in a previous post, a more advanced algorithm could have given us more predictive power. So, we should always keep that in mind. Hope you liked this post. Thanks again and feel free to contact me!

