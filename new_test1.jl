using Pkg; Pkg.activate(@__DIR__);
Pkg.status()

using Oceananigans,
      Oceananigans.Units
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node
using CUDA
using QuadGK
using SpecialFunctions
using CairoMakie
Makie.inline!(true);

Nx, Nz = 500, 200

architecture = GPU()

name = "new_test1"

const H  = 2kilometers
const Lx = 200kilometers

scaler(x)=erf((x-(Lx/4))/(Lx/16))+erf((-x-(Lx/4))/(Lx/16))+3

scale_sum=0
for k in 1:Nx
    global scale_sum=scale_sum+scaler(-(Lx/2)+(k-1)*(Lx/Nx))
end

scaling_factor=Lx/scale_sum

face_array=Array{Float64}(undef,Nx+1)
face_array[1]=-(Lx/2)
for i in 2:(Nx+1)
    face_array[i]=face_array[i-1]+scaling_factor*scaler(-(Lx/2)+(i-1)*(Lx/Nx))
end

new_faces(k)=face_array[trunc(Int,k)]

underlying_grid = RectilinearGrid(architecture,
                                  size = (Nx, Nz),
                                  x = new_faces,
                                  z = (-H, 0),
                                  halo = (4, 4),
                                  topology = (Periodic, Flat, Bounded))
								  
const h0 = 100 # m
const width = 1kilometers
bump(x, y) = - H + h0 * exp(-x^2 / 2width^2)

println("Mount height: ",h0," Mount width: ",width)
println("Grid size: ",Nx,"x",Nz)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bump))

coriolis = FPlane(latitude =-20)


U_mean = 0.05

const forcing_amplitude = U_mean * (coriolis.f)

@inline mean_forcing(x, y, z, t) = forcing_amplitude

Δt = 10

println("Time interval: ",prettytime(Δt))

max_Δt = Δt
free_surface = SplitExplicitFreeSurface(; grid, cfl = 0.7, max_Δt)

const ν = 1000
@inline variable_viscosity(x, y, z, t) = ν*(erf((x-(Lx/4))/(Lx/16))+erf((-x-(Lx/4))/(Lx/16))+2)

horizontal_closure = HorizontalScalarDiffusivity(ν=variable_viscosity, κ=variable_viscosity)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    free_surface = free_surface,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    closure = (horizontal_closure,),
                                    forcing = (v = mean_forcing,))

stop_time = 500hours

simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

using Printf

wall_clock = Ref(time_ns())

function print_progress(sim)

    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹, next Δt: %s\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w), prettytime(sim.Δt))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

b = model.tracers.b
u, v, w = model.velocities

U = Field(Average(u))

u′ = u - U

N² = ∂z(b)

S² = @at (Center, Center, Face) ∂z(u)^2 + ∂z(v)^2

Ri = N² / S²

pressure =  model.pressure.pHY′
@inline bottom_condition(i, j, k, grid) = immersed_peripheral_node(i, j, k-1, grid, Center(), Center(), Center())
bottom_pressure = Field(Average(pressure, condition = bottom_condition, mask = 0.0, dims = 3))

output_interval=15minutes		  

simulation.output_writers[:fields] = JLD2OutputWriter(model, (;N², Ri, u′, u, w, b, pressure,bottom_pressure),
                                                      schedule = TimeInterval(output_interval),
                                                      with_halos = false,
                                                      filename = name,
                                                      overwrite_existing = true)
													  
# Initial conditions
ui(x, y, z) = U_mean

Ni² = (2e-3)*(2e-3)  # [s⁻²] initial buoyancy frequency / stratification
bi(x, y, z) = Ni² * z

println("Stratification: ",sqrt(Ni²))

set!(model, u=ui, b=bi)

froude=sqrt(Ni²)*h0/U_mean

println("Froude number: ",froude)

run!(simulation)	

saved_output_filename = name * ".jld2"

u_t  = FieldTimeSeries(saved_output_filename, "u")
u′_t = FieldTimeSeries(saved_output_filename, "u′")
w_t  = FieldTimeSeries(saved_output_filename, "w")
N²_t = FieldTimeSeries(saved_output_filename, "N²")
bottom_pressure_t = FieldTimeSeries(saved_output_filename, "bottom_pressure")
pressure_t=FieldTimeSeries(saved_output_filename, "pressure")

times = u_t.times

bumpderiv(x)=-( (h0*x) /width^2)*exp(-x^2 / 2width^2) #Computing the derivative of the bump function		
framesize=size(times)[1]

bottom_force=Array{Float64}(undef,framesize,Nx)
for i in 1:framesize
    for j in 1:Nx
        bottom_force[i,j]=(interior(bottom_pressure_t[i])[j])*(quadgk(bumpderiv,new_faces(j),new_faces(j+1),rtol=1e-10)[1])
    end
end

force_t=Array{Float64}(undef,framesize)
for i in 1:framesize
    force_t[i]=0
    for j in 1:Nx
        force_t[i]=force_t[i]+bottom_force[i,j]
    end
end

for i in 1:framesize
	println(force_t[i])
end

xu,  yu,  zu  = nodes(u_t[1])
xw,  yw,  zw  = nodes(w_t[1])
xN², yN², zN² = nodes(N²_t[1])

using Oceananigans.ImmersedBoundaries: mask_immersed_field!

function mask_and_get_interior(φ_t, n)
    mask_immersed_field!(φ_t[n], NaN)
    return interior(φ_t[n], :, 1, :)
end

using Printf

n = Observable(1)

title = @lift @sprintf("t = %1.2f days = %1.2f T2", round(times[$n]/day, digits=2) , round(times[$n]/(12hours), digits=2))

u′n = @lift mask_and_get_interior(u′_t, $n)
wn  = @lift mask_and_get_interior(w_t, $n)
N²n = @lift mask_and_get_interior(N²_t, $n)

axis_kwargs = (xlabel = "x [km]",
               ylabel = "z [m]",
               limits = ((-Lx/2e3, Lx/2e3), (-H, 0)),
               titlesize = 20)

ulim   = 0.5 * maximum(abs, u_t[end])
wlim   = maximum(abs, w_t[end])

fig = Figure(resolution = (700, 900))

ax_u = Axis(fig[2, 1];
            title = "u′-velocity", axis_kwargs...)

ax_w = Axis(fig[3, 1];
            title = "w-velocity", axis_kwargs...)

ax_N² = Axis(fig[4, 1];
             title = "stratification", axis_kwargs...)

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

hm_u = heatmap!(ax_u, xu/1e3, zu, u′n;
                colorrange = (-ulim, ulim),
                colormap = :balance)
Colorbar(fig[2, 2], hm_u)

hm_w = heatmap!(ax_w, xw/1e3, zw, wn;
                colorrange = (-wlim, wlim),
                colormap = :balance)
Colorbar(fig[3, 2], hm_w)

hm_N² = heatmap!(ax_N², xN²/1e3, zN², N²n;
                 colorrange = (0.9Ni², 1.1Ni²),
                 colormap = :thermal)
Colorbar(fig[4, 2], hm_N²)

@info "Making an animation from saved data..."

frames = 1:length(times)

CairoMakie.record(fig, name * ".mp4", frames, framerate=24) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end