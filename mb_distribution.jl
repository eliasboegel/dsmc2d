using Random, Distributions
using Plots

kB = 1.380649e-23
wall_temp = 500
m = 6.63e-26
scale_parameter = sqrt(kB * wall_temp / m)

speed = rand(LocationScale(0, scale_parameter, Chi(3)), 1000000)
histogram(speed, bins=100)