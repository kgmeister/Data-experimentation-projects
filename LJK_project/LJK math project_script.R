library(tidyverse)
library(ggplot2)
library(dplyr)
library(scales)
library(ggpubr)
library(corrplot)


Math_project <- read.csv ("WA3 Math Project.csv")


# Converting character columns to vectors
Math_project$Stream <- factor(Math_project$Stream)
Math_project$Race <- factor(Math_project$Race)
Math_project$Gender <- factor(Math_project$Gender)

# Summary
summary(Math_project)

#Calculating mode for the 3 columns of interest:
cols <- c("Average_Wkday_hrs", "Average_Wkend_hrs", "App_count_opened")

modes <- sapply(Math_project[cols], function(x) {
  as.numeric(names(which.max(table(x))))
})

print(modes)


################################################ Graphs #############################################################

#Dot plot for average weekday hours

ggplot(data = Math_project, aes(x=Average_Wkday_hrs)) +
  geom_dotplot(binwidth = 0.5, binaxis= 'x', stackgroups = TRUE, dotsize = .75, stackratio = 1.2, fill = "#648FFF", method = "histodot") + 
  scale_y_continuous(name = "Student Count") +
  scale_x_continuous(breaks = seq(0, 10, by = 1), name = "Number of Average weekday hours") +
  labs(title = "Stacked Dot Plot of hours spent on using phone") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        axis.line.y = element_blank(),          
        axis.ticks.y = element_blank(),         
        axis.text.y = element_blank())



#Dot plot for average weekend hours

ggplot(data = Math_project, aes(x=Average_Wkend_hrs)) +
  geom_dotplot(binwidth = 0.5, binaxis= 'x', stackgroups = TRUE, dotsize = .75, stackratio = 1.2, fill = "#FE6100", method = "histodot") + 
  scale_y_continuous(name = "Student Count") +
  scale_x_continuous(breaks = seq(0, 10, by = 1), name = "Number of Average weekend hours") +
  labs(title = "Stacked Dot Plot of hours spent on using phone") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        axis.line.y = element_blank(),          
        axis.ticks.y = element_blank(),         
        axis.text.y = element_blank())


#Dot plot for app open count
ggplot(data = Math_project, aes(x=App_count_opened)) +
  geom_dotplot(binwidth = 0.5, binaxis= 'x', stackgroups = TRUE, dotsize = .75, stackratio = 1.2, fill = "#FE6100", method = "histodot") + 
  scale_y_continuous(name = "Student Count") +
  scale_x_continuous(breaks = seq(0, 10, by = 1), name = "Number of App open count") +
  labs(title = "Stacked Dot Plot of App opened count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        axis.line.y = element_blank(),          
        axis.ticks.y = element_blank(),         
        axis.text.y = element_blank())


# Scatterplot for app count vs average hours

Math_project %>%
  pivot_longer(cols = c(Average_Wkday_hrs, Average_Wkend_hrs),
               names_to = "Day_Type",
               values_to = "Hours") %>%
  ggplot(aes(y = Hours, x = App_count_opened, color = Day_Type)) +
  geom_point(alpha = 0.7, size = 4) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ Day_Type, scales = "free_x") +
  scale_color_manual(values = c("#1f77b4", "#ff7f0e")) +
  scale_x_continuous(breaks = seq(0, 10, by = 1)) +
  scale_y_continuous(breaks = seq(0, 10, by = 1)) +
  labs(title = "App Opens vs. Phone usage Hours (Weekday vs. Weekend)",
       y = "Average Hours",
       x = "App Opened Count") +
  theme_minimal() +
    theme(plot.title = element_text(size = 11, hjust = -0.05))



# Scatterplot for unlock count vs average hours

Math_project %>%
  pivot_longer(cols = c(Average_Wkday_hrs, Average_Wkend_hrs),
               names_to = "Day_Type",
               values_to = "Hours") %>%
  ggplot(aes(x = Hours, y = Unlock_count_times, color = Day_Type)) +
  geom_point(alpha = 0.7, size = 4) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ Day_Type, scales = "free_x") +
  scale_color_manual(values = c("#1f77b4", "#ff7f0e")) +
  scale_x_continuous(breaks = seq(0, 10, by = 1)) +
  labs(title = "Phone unlock count vs. Usage Hours (Weekday vs. Weekend)",
       x = "Average Hours",
       y = "Phone unlock Count") +
  theme_minimal() +
  theme(plot.title = element_text(size = 11, hjust = -0.4))


  

#Race vs weekday hours histogram

ggplot(Math_project, aes(x = Average_Wkday_hrs, fill = Race)) +
  geom_histogram(
    binwidth = 1,
    color = "white",
    boundary = 0.5,
    alpha = 0.8,
    position = "identity"
  ) +
  facet_wrap(~ Race, ncol = 2) +  # Separate plots by race
  scale_fill_manual(values = c(
    "Chinese" = "#F28E2B",
    "Malay" = "#8B4513")) +
  scale_x_continuous(breaks = seq(0, 10, by = 1))+
  labs(
    title = "Distribution of Weekday Phone usage by Race",
    x = "Average Weekday Hours",
    y = "Number of Students",
    caption = "Bin width = 1 hour"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none"  
  )


#Race vs weekend hours histogram

ggplot(Math_project, aes(x = Average_Wkend_hrs, fill = Race)) +
  geom_histogram(
    binwidth = 1,
    color = "white",
    boundary = 0.5,
    alpha = 0.8,
    position = "identity"
  ) +
  facet_wrap(~ Race, ncol = 2) +  # Separate plots by race
  scale_fill_manual(values = c(
    "Chinese" = "#F28E2B",
    "Malay" = "#8B4513")) +
  scale_x_continuous(breaks = seq(0, 10, by = 1))+
  labs(
    title = "Distribution of Weekend Phone usage by Race",
    x = "Average Weekday Hours",
    y = "Number of Students",
    caption = "Bin width = 1 hour"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "none"  
  )



###############################################################
###############################################################

#facet scatterplot

ggplot(Math_project, aes(x = App_count_opened, y = Unlock_count_times, color = Gender)) +
  geom_point(alpha = 0.7, size = 3) +
  geom_smooth(method = "lm", se = FALSE) +  # Add trendlines
  facet_wrap(~ Gender, ncol = 2) +          # Separate by stream
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(breaks = seq(0, 10, by = 1)) +
  labs(
    title = "Scatterplots of App Opens vs. Phone Unlocks by Gender",
    x = "App Count Opened",
    y = "Unlock Count Times"
  ) +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(size = 11, face = "bold", hjust = 0.5))
  

#Bubbleplot

ggplot(Math_project, aes(x = Stream, y = App_count_opened, size = Unlock_count_times, color = Stream)) +
  geom_jitter(alpha = 0.7, width = 0.2) +  # Jitter to avoid overplotting
  scale_size_continuous(range = c(2, 10)) + 
  scale_y_continuous(breaks = seq(0, 11, by = 1)) +
  scale_color_manual(
    values = c("#FF355E", "#00CC99", "#FFD700"),
    name = "Stream"                         
  )+# Adjust bubble size range
  labs(
    title = "App Opens vs. Stream (Bubble Size = Unlock Count)",
    x = "Stream",
    y = "App Count Opened"
  ) +
  theme_minimal()

