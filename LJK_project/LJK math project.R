library(tidyverse)
library(ggplot2)
library(dplyr)

setwd("C:/Users/Kenneth/Desktop/LJK_stuff_projects")

Math_project <- read.csv ("WA3 Math Project.csv")


# Converting character columns to vectors
Math_project$Stream <- factor(Math_project$Stream)
Math_project$Race <- factor(Math_project$Race)
Math_project$Gender <- factor(Math_project$Gender)

# Summary
summary(Math_project)

############# Graphs ##############

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












###################################################################################################
# Classic LGBT rainbow colors (vibrant version)
lgbt_colors <- c("#FF0000", # Vibrant red
                    "#FF8C00", # Bright orange
                    "#FFD700", # Gold/yellow
                    "#32CD32", # Lime green
                    "#1E90FF", # Dodger blue
                    "#9932CC") # Dark orchid (purple)

#bar graph
ggplot(data=Math_project, aes(x=Student_ID, y=Average_Wkend_hrs,fill = Average_Wkend_hrs)) +
  geom_bar(stat="identity")+
  scale_fill_gradientn(colours = lgbt_colors) +
  labs(title = "Bar Plot with Vibrant LGBT Rainbow Colors",
       x = "Student_ID",
       y = "Average_weekend_hours") +
  theme_minimal() +
  theme(legend.position = "none")


#line graph
ggplot(Math_project, aes(x = Student_ID, y = Average_Wkend_hrs, color = lgbt_colors)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = lgbt_colors[1:3]) + # Using first 3 colors
  labs(title = "Line Graph with Discrete LGBT Colors",
       x = "Time",
       y = "Value") +
  theme_minimal() +
  theme(legend.position = "bottom")

##############################################################################


# Read and prepare data
data <- read.csv("WA3 Math Project.csv") 

# Define vibrant LGBT rainbow color gradient
rainbow_palette <- c("#FF0000", # Vibrant red
                     "#FF8C00", # Bright orange
                     "#FFD700", # Gold/yellow
                     "#32CD32", # Lime green
                     "#1E90FF", # Dodger blue
                     "#9932CC") # Dark orchid (purple)

# Create dot plot with correct orientation
ggplot(data, aes(x = reorder(Student_ID, Average_Wkend_hrs), 
                 y = Average_Wkend_hrs,
                 color = Average_Wkend_hrs)) +
  geom_point(size = 5, alpha = 0.8) +
  scale_color_gradientn(colors = rainbow_palette,
                        name = "Weekend Hours",
                        guide = guide_colorbar(barwidth = 1, barheight = 10)) +
  labs(title = "Weekend Study Hours by Student",
       subtitle = "Colors represent hours (LGBT rainbow spectrum)",
       x = "Student ID",
       y = "Average Weekend Hours") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        panel.grid.major.x = element_line(color = "grey90"),
        panel.grid.minor.y = element_blank(),
        plot.title = element_text(face = "bold", size = 14),
        legend.position = "right")

