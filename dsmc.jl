# Uncomment to install dependencies on first run
# import Pkg; Pkg.add("Distributions"); Pkg.add("GLMakie");

using Random, Distributions, Statistics
using GLMakie
using Printf

const global kB::Float64 = 1.380649e-23

struct Geometry
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    c_center_x::Float64
    c_center_y::Float64
    c_radius::Float64
    nx::Int64
    ny::Int64
end

struct Gas
    m::Float64 # Molar mass
    T::Float64 # Initial temperature
    sigma::Float64 # Collision cross-section
    n::Float64 # Number density
end

function in_bbox(x::Float64, y::Float64, geom::Geometry)
    return (x >= geom.x_min) && (x <= geom.x_max) && (y >= geom.y_min) && (y <= geom.y_max)
end

function in_cylinder(x::Float64, y::Float64, geom::Geometry)
    return ((x - geom.c_center_x)^2 + (y - geom.c_center_y)^2) < geom.c_radius^2
end

function in_domain(x::Float64, y::Float64, geom::Geometry)
    return in_bbox(x, y, geom) && !in_cylinder(x, y, geom)
end

function specular_reflection(x::Float64, y::Float64, dxdt::Float64, dydt::Float64, normal_x::Float64, normal_y::Float64, dt::Float64)
    dxdt_new = dxdt - 2 * (dxdt * normal_x + dydt * normal_y) * normal_x
    dydt_new = dydt - 2 * (dxdt * normal_x + dydt * normal_y) * normal_y
    x = x + (dxdt + dxdt_new) / 2 * dt
    y = y + (dydt + dydt_new) / 2 * dt
    dxdt = dxdt_new
    dydt = dydt_new
    return x, y, dxdt, dydt
end

function diffuse_reflection(x::Float64, y::Float64, dxdt::Float64, dydt::Float64, normal_x::Float64, normal_y::Float64, wall_temp::Float64, dt::Float64)
    # Sample angle with a cos(x) distribution on [0, pi/2], produces angle between -90 and 90 degrees off of normal
    sin_angle = rand(Uniform(-1, 1))
    cos_angle = sqrt(1 - sin_angle^2)

    # Compute new velocity vector sampled from Lambert law
    # Sample from Maxwell-Boltzmann distribution for speed (not velocity!), equal to Chi-distribution with k=3 and scale parameter sqrt(kB * T / m)
    scale_parameter = sqrt(kB * wall_temp / gas.m) # Set standard deviation of Maxwellian using wall temperature as reference temperature

    speed_new = rand(LocationScale(0, scale_parameter, Chi(3))) # Sample from Maxwellian and convert to speed instead of velocity
    dxdt_new = (cos_angle * normal_x - sin_angle * normal_y) * speed_new
    dydt_new = (sin_angle * normal_x + cos_angle * normal_y) * speed_new

    # Propagate particle using new velocity, assuming wall collision is half dt away
    x = x + (dxdt + dxdt_new) / 2 * dt
    y = y + (dydt + dydt_new) / 2 * dt
    dxdt = dxdt_new
    dydt = dydt_new

    return x, y, dxdt, dydt
end

function push_particles(geom::Geometry, x::Vector{Float64}, y::Vector{Float64}, dxdt::Vector{Float64}, dydt::Vector{Float64}, F_n::Float64, dt::Float64)
    Threads.@threads for i in eachindex(x)
        x_new = x[i] + dxdt[i] * dt
        y_new = y[i] + dydt[i] * dt

        if y_new < geom.y_min # Case: Particle moved through symmetry boundary -> symmetry boundary collision
            x[i], y[i], dxdt[i], dydt[i] = specular_reflection(x[i], y[i], dxdt[i], dydt[i], 0.0, 1.0, dt)
        elseif ((x_new - geom.c_center_x)^2 + (y_new - geom.c_center_y)^2) < geom.c_radius^2 # Case: Particle moved into cylinder -> cylinder boundary collision
            # At small dt, can approximate normal at mean of x,y and x_new,y_new instead of exact intersection
            # Then, approximate cylinder wall as small segment of straight wall at "half-way distance" with computed normal
            mean_x = (x_new + x[i]) / 2
            mean_y = (y_new + y[i]) / 2
            d = sqrt((mean_x - geom.c_center_x)^2 + (mean_y - geom.c_center_y)^2)
            normal_x = (mean_x - geom.c_center_x) / d
            normal_y = (mean_y - geom.c_center_y) / d

            x[i], y[i], dxdt[i], dydt[i] = diffuse_reflection(x[i], y[i], dxdt[i], dydt[i], normal_x, normal_y, 500.0, dt)
        else # Case: Particle did not move across any boundary -> normal update of position
            x[i] = x_new
            y[i] = y_new
        end

        # Resample if particle leaves domain
        if !in_domain(x[i], y[i], geom)
            stddev = sqrt(kB * gas.T / gas.m)
            r = rand() # Random number to decide whether particle is resampled at left boundary or at top boundary
            # To approximate flux into the domain, integrate x*VDF where VDF is the 1D Maxwellian in free stream
            # For y-direction flux, determine y-direction into-domain bulk velocity from symmetric VDF (used for y-direction flux): int_0^inf c*x*exp(-a*x^2) dx = c/(2a)
            # For x-direction flux, simply use freestream bulk velocity of 2634 m/s
            a = gas.m / (2 * kB * gas.T)
            c = sqrt(gas.m / (2 * pi * kB * gas.T))
            flux_left = (geom.y_max - geom.y_min) * gas.n * 2634 / F_n # Approximate flux (l * n * ux / F_n) into domain across left boundary using free stream bulk velocity
            flux_top = (geom.x_max - geom.x_min) * gas.n * c / (2 * a) / F_n # Approximate flux (l * n * uy / F_n) into domain across top boundary using y-direction Maxwellian

            if r < flux_top / (flux_left + flux_top) # Probability to choose top boundary is length of top boundary divided by total length of left and top boundary combined
                x[i] = rand(Uniform(geom.x_min, geom.x_max))
                y[i] = geom.y_max
                dxdt[i] = rand(Normal(2634.0, stddev))
                dydt[i] = -abs(rand(Normal(0.0, stddev)))
            else # If top boundary was not chosen, then choose left boundary
                x[i] = geom.x_min
                y[i] = rand(Uniform(geom.y_min, geom.y_max))
                dxdt[i] = rand(Normal(2634.0, stddev))
                dydt[i] = rand(Normal(0.0, stddev))
            end
        end
    end
end

function vhs(dxdt1::Float64, dydt1::Float64, dxdt2::Float64, dydt2::Float64)
    scattering_angle = 2 * pi * rand() # Select random scattering angle
    dxdt_cr = dxdt1 - dxdt2
    dydt_cr = dydt1 - dydt2
    cr_mag = sqrt(dxdt_cr^2 + dydt_cr^2) # Determine relative speed
    dxdt_cr = cr_mag * cos(scattering_angle)
    dydt_cr = cr_mag * sin(scattering_angle)

    dxdt_m = (dxdt1 + dxdt2) / 2
    dydt_m = (dydt1 + dydt2) / 2
    dxdt1 = dxdt_m + 0.5 * dxdt_cr
    dydt1 = dydt_m + 0.5 * dydt_cr
    dxdt2 = dxdt_m - 0.5 * dxdt_cr
    dydt2 = dydt_m - 0.5 * dydt_cr
    return dxdt1, dydt1, dxdt2, dydt2
end

function collide_particles(geom::Geometry, cell_fractions, x::Vector{Float64}, y::Vector{Float64}, dxdt::Vector{Float64}, dydt::Vector{Float64}, F_n::Float64, dt::Float64, N_c_carry::Vector{Float64})
    # Determine cell index for every particle: i = ix + iy*nx
    hx::Float64 = (geom.x_max - geom.x_min) / geom.nx
    hy::Float64 = (geom.y_max - geom.y_min) / geom.ny
    cell_indices_i::Vector{Int64} = trunc.(Int64, (x .- geom.x_min) ./ hx)
    cell_indices_j::Vector{Int64} = trunc.(Int64, (y .- geom.y_min) ./ hy)
    cell_indices::Vector{Int64} = cell_indices_i + cell_indices_j .* nx .+ 1

    # Sort particles by cell index
    i_s = sortperm(cell_indices)
    cell_indices_s = cell_indices[i_s]
    cell_indices_i_s = cell_indices_i[i_s]
    cell_indices_j_s = cell_indices_j[i_s]
    x_s = x[i_s]
    y_s = y[i_s]
    dxdt_s = dxdt[i_s]
    dydt_s = dydt[i_s]
    Threads.@threads for cell_index in 1:(geom.nx*geom.ny) # Iterate over all possible cells
        cell_range = searchsorted(cell_indices_s, cell_index) # Find indices in sorted cell indices which belong to this cell

        # Select particles in cell
        i_s_cell = i_s[cell_range]
        x_cell = x_s[cell_range]
        y_cell = y_s[cell_range]
        dxdt_cell = dxdt_s[cell_range]
        dydt_cell = dydt_s[cell_range]

        # Compute collision pairs
        N = length(x_cell) # Number of particles in cell

        if N > 1 # Collisions only possible if at least 2 particles exist in cell
            pairs = [(i, j) for i = 1:N-1 for j = i+1:N] # Create list of all unique unordered combinations of particles within cell
            sigma_rel_speeds = [gas.sigma * sqrt((dxdt_cell[pairs[i][1]] - dxdt_cell[pairs[i][2]])^2 + (dydt_cell[pairs[i][1]] - dydt_cell[pairs[i][2]])^2) for i = 1:length(pairs)]
            max_sigma_rel_speed = maximum(sigma_rel_speeds)

            cell_idx_i = cell_indices_i_s[cell_range][1] + 1
            cell_idx_j = cell_indices_j_s[cell_range][1] + 1
            V_c::Float64 = hx * hy * cell_fractions[cell_idx_i, cell_idx_j] # Cell volume

            N_c = 0.5 * N * (N - 1) * F_n * max_sigma_rel_speed * dt / V_c # Compute number of particles to be collided
            N_c_f, N_c_i = modf(N_c) # Split N_c into integer and fractional part
            N_c_add_f, N_c_add_i = modf(N_c_f + N_c_carry[cell_index]) # Split total current fractional part into integer and fractional part
            N_c_i = N_c_i + N_c_add_i # Add integer contribution of carry-over + current fractional part
            N_c_carry[cell_index] = N_c_add_f # Carry over new total fractional part
            pair_p = sigma_rel_speeds ./ max_sigma_rel_speed # Compute collision probability for each pair

            for i in 1:N_c_i
                pair_idx = rand(1:length(pairs))
                r = rand()
                if r < pair_p[pair_idx]
                    idx1 = i_s_cell[pairs[pair_idx][1]]
                    idx2 = i_s_cell[pairs[pair_idx][2]]
                    dxdt[idx1], dydt[idx1], dxdt[idx2], dydt[idx2] = vhs(dxdt[idx1], dydt[idx1], dxdt[idx2], dydt[idx2])
                end
            end

        end
    end
end

function compute_cell_fractions(geom::Geometry)
    # Computes the fraction of a cell that is inside the domain (necessary for the non-axis aligned boundary of the cylinder with the immersed grid)
    cell_fraction = zeros(geom.nx, geom.ny)
    hx = (geom.x_max - geom.x_min) / geom.nx
    hy = (geom.y_max - geom.y_min) / geom.ny
    for i in 1:geom.nx
        x_cell_min = geom.x_min + (i - 1) * hx
        x_cell_max = geom.x_min + i * hx
        for j in 1:geom.ny
            y_cell_min = geom.y_min + (j - 1) * hy
            y_cell_max = geom.y_min + j * hy

            if in_domain(x_cell_min, y_cell_min, geom) && in_domain(x_cell_max, y_cell_min, geom) && in_domain(x_cell_min, y_cell_max, geom) && in_domain(x_cell_max, y_cell_max, geom) # All corner points of cell are in domain
                cell_fraction[i, j] = 1
            elseif !in_domain(x_cell_min, y_cell_min, geom) && !in_domain(x_cell_max, y_cell_min, geom) && !in_domain(x_cell_min, y_cell_max, geom) && !in_domain(x_cell_max, y_cell_max, geom) # All corner points of cell are not in domain
                cell_fraction[i, j] = 0
            else
                # Determine cell fraction in Monte-Carlo way
                n_samples = 1000
                for k in 1:n_samples
                    x_s = rand(Uniform(x_cell_min, x_cell_max))
                    y_s = rand(Uniform(y_cell_min, y_cell_max))
                    cell_fraction[i, j] += in_domain(x_s, y_s, geom)
                end
                cell_fraction[i, j] = cell_fraction[i, j] / n_samples
            end
        end
    end
    return cell_fraction
end

function eval_fields(cell_fractions, x::Vector{Float64}, y::Vector{Float64}, dxdt::Vector{Float64}, dydt::Vector{Float64}, F_n::Float64)
    number_density = zeros(size(cell_fractions))
    u_x = zeros(size(cell_fractions))
    u_y = zeros(size(cell_fractions))
    T = zeros(size(cell_fractions))

    # Determine cell index for every particle: i = ix + iy*nx
    hx::Float64 = (geom.x_max - geom.x_min) / geom.nx
    hy::Float64 = (geom.y_max - geom.y_min) / geom.ny
    cell_indices_i::Vector{Int64} = trunc.(Int64, (x .- geom.x_min) ./ hx)
    cell_indices_j::Vector{Int64} = trunc.(Int64, (y .- geom.y_min) ./ hy)
    cell_indices::Vector{Int64} = cell_indices_i + cell_indices_j .* nx .+ 1

    # Sort particles by cell index
    i_s = sortperm(cell_indices)
    cell_indices_s = cell_indices[i_s]
    cell_indices_i_s = cell_indices_i[i_s]
    cell_indices_j_s = cell_indices_j[i_s]
    x_s = x[i_s]
    y_s = y[i_s]
    dxdt_s = dxdt[i_s]
    dydt_s = dydt[i_s]

    Threads.@threads for cell_index in 1:(geom.nx*geom.ny) # Iterate over all possible cells
        cell_range = searchsorted(cell_indices_s, cell_index) # Find indices in sorted cell indices which belong to this cell

        # Select particles in cell
        i_s_cell = i_s[cell_range]
        x_cell = x_s[cell_range]
        y_cell = y_s[cell_range]
        dxdt_cell = dxdt_s[cell_range]
        dydt_cell = dydt_s[cell_range]

        N = length(i_s_cell)
        if N > 0 # Proceed only if any particles are in cell
            cell_idx_i = cell_indices_i_s[cell_range][1] + 1
            cell_idx_j = cell_indices_j_s[cell_range][1] + 1
            cell_volume_mult = cell_fractions[cell_idx_i, cell_idx_j] == 0 ? 0 : 1 / (hx * hy * cell_fractions[cell_idx_i, cell_idx_j]) # Inverse of true cell volume (takes reduced cell volume into account for cut cells)

            number_density[cell_idx_i, cell_idx_j] = N * F_n * cell_volume_mult
            u_x[cell_idx_i, cell_idx_j] = mean(dxdt_cell)
            u_y[cell_idx_i, cell_idx_j] = mean(dydt_cell)
            T[cell_idx_i, cell_idx_j] = gas.m / (3 * kB) * mean((dxdt_cell .- u_x[cell_idx_i, cell_idx_j]) .^ 2 .+ (dydt_cell .- u_y[cell_idx_i, cell_idx_j]) .^ 2)
        end
    end

    return number_density, u_x, u_y, T
end

function plot_field(geom::Geometry, field, label, filename; logscale=false)
    hx = (geom.x_max - geom.x_min) / geom.nx
    hy = (geom.y_max - geom.y_min) / geom.ny
    fig = Figure(backgroundcolor=:white)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    ax.xlabel = "x"
    ax.ylabel = "y"
    xlims!(ax, geom.x_min, geom.x_max)
    ylims!(ax, geom.y_min, geom.y_max)
    if logscale
        im = heatmap!(geom.x_min:hx:geom.x_max, geom.y_min:hy:geom.y_max, field, colormap=:jet1, colorscale=log10)
    else
        im = heatmap!(geom.x_min:hx:geom.x_max, geom.y_min:hy:geom.y_max, field, colormap=:jet1)
    end
    Colorbar(fig[2, 1], im, vertical=false, minorticksvisible=true, label=label)
    mesh!(Circle(Point2f(geom.c_center_x, geom.c_center_y), geom.c_radius), color="black")
    path = "img/" * filename * ".png"
    println(path)
    save(path, fig, px_per_unit=3)
    display(GLMakie.Screen(), fig)
end

function plot_particles(x::Vector{Float64}, y::Vector{Float64})
    obs_x = Observable(x)
    obs_y = Observable(y)
    data = @lift(Point2f.([$obs_x; $obs_x], [$obs_y; -$obs_y])) # Full domain (incl. symmetry)
    # data = @lift(Point2f.($obs_x, $obs_y)) # Half domain
    fig = Figure(backgroundcolor=:white)
    display(GLMakie.Screen(), fig)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    ax.xlabel = "x"
    ax.ylabel = "y"
    xlims!(ax, geom.x_min, geom.x_max)
    ylims!(ax, -geom.y_max, geom.y_max)
    scatter!(data, color=:red, markersize=2)
    mesh!(Circle(Point2f(geom.c_center_x, geom.c_center_y), geom.c_radius), color="black")

    return obs_x, obs_y
end

function init_particles(geom::Geometry, np::Int64)
    # Initialize particles
    x = Vector{Float64}(undef, np)
    y = Vector{Float64}(undef, np)
    dxdt = Vector{Float64}(undef, np)
    dydt = Vector{Float64}(undef, np)

    # Uniformly distribute particles across domain
    stddev = sqrt(kB * gas.T / gas.m) # Set velocities as Maxwellian with temperature and bulk velocities (drifting Maxwellian distribution)
    for i in eachindex(x)
        x[i] = rand(Uniform(geom.x_min, geom.x_max))
        y[i] = rand(Uniform(geom.y_min, geom.y_max))
        while !in_domain(x[i], y[i], geom)
            x[i] = rand(Uniform(geom.x_min, geom.x_max))
            y[i] = rand(Uniform(geom.y_min, geom.y_max))
        end
        dxdt[i] = rand(Normal(2634.0, stddev))
        dydt[i] = rand(Normal(0.0, stddev))
    end
    return x, y, dxdt, dydt
end

function dsmc(geom::Geometry, gas::Gas, nc::Int64, t_final::Float64, t_sample::Float64, dt::Float64)
    hx = (geom.x_max - geom.x_min) / geom.nx
    hy = (geom.y_max - geom.y_min) / geom.ny
    cell_fractions = compute_cell_fractions(geom)

    # Compute superparticle count and superparticle weight
    V_d::Float64 = (geom.x_max - geom.x_min) * (geom.y_max - geom.y_min) - 0.5 * pi * geom.c_radius^2 # Domain volume
    np::Int64 = nc * geom.nx * geom.ny
    F_n::Float64 = V_d * gas.n / np # Ratio of real molecules vs superparticles

    println("Mean Free Path (MFP): ", 1 / (gas.sigma * gas.n), ", cell: ", min(hx, hy), ", Cell/MFP: ", min(hx, hy) * (gas.sigma * gas.n))
    println("Mean Free Time (MFT): ", 1 / (gas.sigma * gas.n) / 2634.0, ", dt: ", dt, ", dt/MFT: ", dt * 2634.0 * (gas.sigma * gas.n))

    x, y, dxdt, dydt = init_particles(geom, np)
    obs_x, obs_y = plot_particles(x, y)

    N_c_carry = zeros(geom.nx * geom.ny) # Vector to store fractional carry-overs of N_c for each cell

    number_density = zeros(geom.nx, geom.ny)
    bulk_velocity_x = zeros(geom.nx, geom.ny)
    bulk_velocity_y = zeros(geom.nx, geom.ny)
    kin_temperature = zeros(geom.nx, geom.ny)
    sample_weight::Float64 = 1 / trunc(Int64, (t_final - t_sample) / dt)

    t::Float64 = 0.0
    while t < t_final
        t += dt
        @printf "t = %.2f µs\n" t * 1e6

        push_particles(geom, x, y, dxdt, dydt, F_n, dt)
        collide_particles(geom, cell_fractions, x, y, dxdt, dydt, F_n, dt, N_c_carry)
        obs_x[], obs_y[] = x, y # Update particle plot

        if t > t_sample
            number_density_it, bulk_velocity_x_it, bulk_velocity_y_it, kin_temperature_it = eval_fields(cell_fractions, x, y, dxdt, dydt, F_n)
            number_density .+= number_density_it * sample_weight
            bulk_velocity_x .+= bulk_velocity_x_it * sample_weight
            bulk_velocity_y .+= bulk_velocity_y_it * sample_weight
            kin_temperature .+= kin_temperature_it * sample_weight
        end

    end
    replace!(number_density, 0 => NaN)
    replace!(bulk_velocity_x, 0 => NaN)
    replace!(bulk_velocity_y, 0 => NaN)
    replace!(kin_temperature, 0 => NaN)
    bulk_speed = sqrt.(bulk_velocity_x .^ 2 .+ bulk_velocity_y .^ 2)
    mach_number = bulk_speed ./ sqrt.((5 / 3 * kB / gas.m) .* kin_temperature)
    plot_field(geom, number_density, rich("Number density [m", superscript("-3"), "]"), "n", logscale=true)
    plot_field(geom, bulk_speed, rich("Bulk velocity [ms", superscript("-1"), "]"), "u")
    plot_field(geom, mach_number, rich("Mach number [-]"), "M")
    plot_field(geom, kin_temperature, rich("Kinetic Temperature [K]"), "T")
end

x_min, x_max = -0.2, 0.65
y_min, y_max = 0.0, 0.4
nx = 300
ny = trunc(Int64, nx * (y_max - y_min) / (x_max - x_min))
print("nx: ", nx, ", ny: ", ny)
geom = Geometry(x_min, x_max, y_min, y_max, 0.1524, 0, 0.1524, nx, ny) # Box with cylinder in the middle
gas = Gas(6.63e-26, 200, pi * (2 * 3.595e-10 / 2)^2, 4.247e20) # Argon at 200 K, 4.247e20 number density

dsmc(geom, gas, 10, 17.5e-4, 7.5e-4, 0.5 * 0.72e-6)