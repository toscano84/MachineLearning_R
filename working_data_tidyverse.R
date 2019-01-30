# Working with Data in the Tidyverse
# readr
library(tidyverse)
library(skimr)
library(janitor)

# load data
bakers <- read_csv("bakeoff.csv",
                   na = c("", "NA", "UNKNOWN"))
View(bakers)

# check data with skim
skim(bakers)

bakers %>%
  skim(us_season)

bakers %>% 
  skim() %>%
  summary()

#----counting----#
# distinct
bakers %>%
  distinct(us_season, series)

# count
bakers %>%
  count(us_season, series) %>%
  mutate(prop_bakers = n/sum(n)) # the output is the same as the one below

bakers %>%
  group_by(us_season, series) %>%
  summarise(n = n()) %>%
  ungroup() %>%
  mutate(prop_bakers = n/sum(n))

bakers %>%
  count(us_season, series) %>%
  count(us_season)

bakers %>%
  count(us_season == 2)

ggplot(bakers, aes(x = episode)) +
  geom_bar() +
  facet_wrap(~series)


#--Cast column types--#
# col_types
desserts <- read_csv("desserts.csv",
                     col_types = cols(
                       technical = col_number()))

View(desserts)

# check problems() in parsing
problems(desserts) # it could not parse N/A

# attribute N/A to na in read_csv
desserts <- read_csv("desserts.csv",
                     col_types = cols(technical = col_number()),
                                      na = c("", "NA", "N/A"))

# parse_date
desserts <- read_csv("desserts.csv",
                     col_types = cols(technical = col_number(),
                                      uk_airdate = col_date(format =
                                                              "%d %B %Y")),
                     na = c("", "NA", "N/A"))


# parse_factor

desserts <- read_csv("desserts.csv",
                     col_types = cols(technical = col_number(),
                                      uk_airdate = 
                                        col_date(format = "%d %B %Y"),
                                    result = col_factor(levels = NULL)),
                     na = c("", "NA", "N/A"))

#---back to dplyr---#
# recode
desserts %>%
  count(nut, sort = TRUE) %>%
  mutate(nut_type = recode(nut, 
                           "filbert" = "hazelnut",
                           "no nut" = "NA_character_"))

desserts %>%
  mutate(tech_win = recode(technical, `1` = 1,
                           .default = 0)) %>%
  glimpse()

# select 
desserts %>%
  select(baker, everything(), -ends_with("airdate"))

# data cleaning
messy <- read_csv("messy_ratings2.csv")
View(messy)

messy %>%
  clean_names("snake")

# back to select
messy2 <- messy %>%
  select(viewers_ =ends_with("7day")) 
View(messy2)

#---------3 Chapter - Tidy Data---#
# gahter, spread, unite, separate
messy_tidy <- messy %>%
  gather(episode, viewers, -series,
         factor_key = TRUE, na.rm = TRUE) %>%
  arrange(series, episode) %>%
  mutate(episode_count = row_number()) %>%
  separate(episode, into = "episode", sep = "_", extra = "drop") %>%
  mutate(episode = parse_number(episode))
View(messy_tidy)

ggplot(messy_tidy, aes(x = episode, y = viewers, 
                         group = series, color = series)) +
  geom_line() +
  facet_wrap(~series) +
  guides(color = FALSE) +
  theme_minimal()

messy_ends <- messy %>%
  select(ends_with("e*_7day")) 

# wt argument in count
View(messy_tidy)
messy_spread <- messy_tidy %>%
  count(series, episode, wt = viewers, sort = TRUE) %>%
  spread(episode, n, sep = "_")
View(messy_spread)
#-------forcats-----#

