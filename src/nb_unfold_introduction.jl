### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 81ca9393-5d98-4d85-96fc-4725787aed18
begin
	using PlutoUI  # For better interactivity
	using Plots  # Native Julia plotting
	using DSP  # Digital Signal Processing
	using DataFrames  # Tidy Dataframes
	using Unfold  # EEG Deconvolution
	using StatsModels  # Make your own Models
	using StatsBase
	using StatsPlots
	using Random  # Julias Random Generator
	using Distributions # Additional Distributions to draw from
	using PlutoUI.ExperimentalLayout: vbox, hbox, Div

	# Table of contents; Depending on your screen size this one and/or the
	# interactive slider box might be hidden. Feel free to comment out the
	# toc and change line 5 to "top: 1rem;" in the next (hidden) cell.
	#PlutoUI.TableOfContents(aside=true, depth=1)
	
end

# ╔═╡ 3e17c9f0-a453-11ec-2206-0bfea70f79d3
md"""
# Unfold Tutorial

"""

# ╔═╡ ccd268ce-dfed-4dd2-8cc2-1d9a39c0e67d
md"""
## About this Tutorial
"""

# ╔═╡ 18bfbf75-5e8d-4bbf-880f-589cbac1999e
md"""
This tutorial is meant as an easy introduction to using the Unfold toolbox/ package and how to correct for overlap. It is written as a Pluto notebook so you don't have to worry about coding anything yourself and get a basic understanding of Unfold and the underlying method.

It is **not** meant to show you the full capabilities of Unfold. For this, please refer to the official [Julia](https://unfoldtoolbox.github.io/Unfold.jl/dev/) or [MATLAB](https://www.unfoldtoolbox.org/overview.html) documentation. Additionally, you should read the [original Unfold paper](https://peerj.com/articles/7838/#p-28).

If you are not familiar with Pluto, it is basically a Julia-specific Jupyter notebook. The notable difference is that variables are automatically shared and updated across all cells.

Lastly, this Notebook is a work in progress and might change in the future. Any feedback is welcome.
"""

# ╔═╡ 09beb045-a5d2-4472-8543-a95e6136e2a0
md"""
!!! note \"OBS!!!\" 
	Occasionally, you will see blue "OBS" boxes. "OBS" is the Norwegian word that combines the English words "note" and "attention". These boxes are meant to point out specific caveats to which you should pay special attention when doing your own analysis with Unfold in the future.
"""

# ╔═╡ 1c69a368-8fe7-4fe2-8a94-562f06f38bfd
md"""
!!! warning \"Question\" 
	You will also stumble upon such yellow "Question" boxes, which are meant to get you thinking. 
"""

# ╔═╡ f0d58bb0-f673-448d-851d-0adb718d5cfe
md"""
!!! tip \"Tips\" 
	Green boxes are meant to give you tips for your own analysis and might point you towards additional resources.
"""

# ╔═╡ 9697844d-3da2-4b3f-811b-eeb075b90fdd
md"""
Acknowledgement goes to Luis Lips for parts of the code as well as to Francesca Zermiani and Judith Schepers for valuable feedback on early versions of the tutorial.
"""

# ╔═╡ 32303686-90db-4abb-ba87-57a58f93af1d
md"""
# Setting up Dependencies
"""

# ╔═╡ d2633e8f-dd86-453b-ac83-61c8990f2723
md"""
# Simulating EEG

To evaluate and show what Unfold can do we will use a trick from methods development research. In short, we will simulate a response kernel multiple times to construct an artificial EEG signal with known ground truth. 
"""

# ╔═╡ c84371c6-9bfb-4632-b641-43612d41023b
md"""
!!! warning \"Question\" 
	Why do we simulate an EEG? Couldn't we just use some real EEG data?
"""

# ╔═╡ 4c48a3ac-2fd2-4462-a655-559e69c4a18b
md"""
## The response Kernel
"""

# ╔═╡ 1ae08d84-de51-439c-a580-4e65c45a1dc8
md"""
First, we will need a ground truth (aka response kernel), which will represent the ERP we want to later estimate. For our simulation, we will use two different kernels which which represent two different experimental stimuli/ conditions (or simply a stimulus and a response) in a real EEG experiment.
"""

# ╔═╡ bf17e5d8-afcf-4537-a25f-0bc3c7389887

# Boxcar function
H(t) = 1. * (sign(t) + 1);


# ╔═╡ 61928269-a3a7-41b6-beb3-0b5acfac6299
begin
	function_A_selection = md"""Choose the response function for stimulus A: $(
		@bind selection₁ Select(
			[1=>"Function1", 2=> "Function2", 3=> "Function3", 4=> "Function4"],
		default=1))"""
end

# ╔═╡ bfc4c110-4e16-4473-b898-0d0f6d4fd21f
begin
	if selection₁ == 1
		Markdown.parse("\$ERP_A(t) = -5(t-b)ℯ^{-0.5(t-b)^2}\$")
	elseif selection₁ == 2
		Markdown.parse("\$ERP_A(t) = 2.5ℯ^{-(t-b)^2}\$")
	elseif selection₁ == 3
		Markdown.parse("\$ERP_A(t) = H(t-1)-H(t-b)\$ \
		with \$H(X)\$ := Heaviside Step Function")
	elseif selection₁ == 4
		Markdown.parse("\$(1/(abs(1/10)*sqrt(π)))ℯ^-((t-b)/(1/10))^2\$")
	end
end

# ╔═╡ 7a6ed8c9-c9cc-4651-8475-ce3e25bce26c
let 
	md"""Change the value of b $(@bind b Slider(4:0.5:6, default=4, show_value=true))"""
end

# ╔═╡ 658ae8ea-061d-4fa2-af6a-690459b40849
begin
	if selection₁ == 1
		erp_A(t) = -5(t-b)ℯ^ -0.5(t-b)^2;
		l_A = Markdown.parse("\$ERP_A(t) = -5(t-$b)ℯ^{-0.5(t-$b)^2}\$")
	elseif selection₁ == 2
		erp_A(t) = 2.5ℯ^(-(t-b)^2);
		l_A = Markdown.parse("\$ERP_A(t) = 2.5ℯ^{-(t-$b)^2}\$")
	elseif selection₁ == 3
		erp_A(t) = H(t-1) - H(t-b)
		l_A = Markdown.parse("\$ERP_A(t) = H(t-1)-H(t-$b)\$ \
		with \$H(X)\$ := Heaviside Step Function")
	elseif selection₁ == 4
		erp_A(t) = (1/(abs(1/10)*sqrt(π)))ℯ^-((t-b)/(1/10))^2
		l_A = Markdown.parse("\$(1/(abs(1/10)*sqrt(π)))ℯ^-((t-$b)/(1/10))^2\$")
	end
end

# ╔═╡ f8b9ab1a-ff14-4e6e-aa8d-e1fb2a49191a
begin
	kernelA = plot(erp_A, xlims=(-2, 10), ylims=(-5,5), legend=false, linecolor=:orange, size=(600,200))
	vline!([0], linestyle=:dash, linecolor=:black)
end

# ╔═╡ d3f9adc8-9e3f-4d68-bf1a-5fc526a0cd40
begin
function_B_selection = md"""Choose the response function for stimulus B: $(
	@bind selection₂ Select(
		[1=>"Function1", 2=>"Function2", 3=> "Function3", 4=> "Function4"],
		default=2))"""
end

# ╔═╡ 339ecc3a-d5cf-452d-b923-1e49e01839ed
if selection₂ == 1
	Markdown.parse("\$ERP_A(t) = -5(t-d)ℯ^{-0.5(t-d)^2}\$")
elseif selection₂ == 2
	Markdown.parse("\$ERP_B(t) = 2.5ℯ^{-(t-d)^2}\$")
elseif selection₂ == 3
	Markdown.parse("\$ERP_B(t) = H(t-1)-H(t-d)\$ \
	with \$H(X)\$ := Heaviside Step Function")
elseif selection₂ ==4
	Markdown.parse("\$(1/(abs(1/10)*sqrt(π)))ℯ^-((t-b)/(1/10))^2\$")
end

# ╔═╡ b05ca432-e0d6-4790-9dfd-ab303ec9abc7
let
	md"""Change the value of d $(@bind d Slider(3:0.5:6, default=3, show_value=true))"""
end

# ╔═╡ 38e57913-991c-4afe-8895-3901eceb9783
if selection₂ == 1
	erp_B(t) = -5(t-d)ℯ^ -0.5(t-d)^2;
	l_B = Markdown.parse("\$ERP_A(t) = -5(t-$d)ℯ^{-0.5(t-$d)^2}\$")
elseif selection₂ == 2
	erp_B(t) = 2.5ℯ^(-(t-d)^2);
	l_B = Markdown.parse("\$ERP_B(t) = 2.5ℯ^{-(t-$d)^2}\$")
elseif selection₂ == 3
	erp_B(t) = H(t-1) - H(t-d)
	l_B = Markdown.parse("\$ERP_B(t) = H(t-1)-H(t-$d)\$ \
	with \$H(X)\$ := Heaviside Step Function")
elseif selection₂ ==4
	erp_B(t) = (1/(abs(1/8)*sqrt(π)))ℯ^-((t-d)/(1/8))^2
	l_B = Markdown.parse("\$(1/(abs(1/10)*sqrt(π)))ℯ^-((t-$b)/(1/10))^2\$")
end

# ╔═╡ 74315375-2094-4779-826a-30b7b980d659
begin
	plot(erp_B, xlims=(-2, 10), ylims=(-5,5), linecolor=:deepskyblue, legend=false, size=(600,200))
	vline!([0], linestyle=:dash, linecolor=:black)
end

# ╔═╡ 7de9ee9d-6176-4923-9667-ded47146862e
md"""
## Event Onsets
"""

# ╔═╡ 7e1eb3ae-829b-419b-8d7f-b08174419b19
md"""
Next, we will need event onsets for our kernels. The exact onsets in an EEG experiment depend on your setup. In our simulation, we can make life easy for us, however, and just simulate 300 onsets for stimuli A and B respectively.

We can either visualize these as onsets over time:
"""

# ╔═╡ 81b7896b-054d-4827-b9b5-a983f39ad674
md"""
Or show how our kernels actually map onto these onsets (top), and how the combined continuous signal looks like (bottom):
"""

# ╔═╡ 7aeeb225-00eb-4197-b360-8d44e6033375
md"""
!!! warning \"Question\" 
	Do these onsets make sense to you? Why/ Why not?
"""

# ╔═╡ d4b54556-f14b-4f31-bba3-e71b077d32ee
md"""
## Noise
"""

# ╔═╡ b58dadff-334c-46e8-9135-891d5d9f33d2
md"""
Lastly, to bring the signal closer to reality we can add some (Gaussian) noise:
"""

# ╔═╡ 7e42a173-4efd-4d0f-8044-251237416514
begin	
	noise_slider = md"""Change noise: σ = $(@bind σ Slider(0:0.2:2, default=0, 
		show_value=true))"""
end

# ╔═╡ e086d4c4-a732-4f7d-a430-905190f3b12e
md"""
# Using the Unfold Toolbox
"""

# ╔═╡ 56c9a93f-4b2a-4891-9a1c-21e57b557911
md"""
## Data Conversion
"""

# ╔═╡ 4f973b6e-f740-422f-a66b-98a46cc97790
md"""
To use Unfold we first need to bring our data into a format that Unfold can actually use.

For Julia this means we end up with two variables:

```data```: containing the continuous signal of size channel x sample-points

```events```: A table containing information about stimulus events and onsets. See below for our example.
"""

# ╔═╡ fdc2a51f-5c75-43ef-9384-0b6521072548
md"""
!!! tip \"Data Structure\" 
	If you use Matlab you can find a handy cheat sheet here: [Unfold Cheatsheet](https://github.com/ReneSkukies/PlutoNotebooks/blob/main/assets/CheatSheet_draft.pdf)
"""

# ╔═╡ 47f80bf2-7c90-4aeb-ac4c-1db2a812285f
md"""
However, we further need to cut our data into epochs. Luckily, Unfold can do that for us if we provide it with the right information.\
Apart from the ```data``` and ```events``` variable we also need information about:\
```τ```: Timewindow around the epoch.\
```sfreq```: Sampling frequency.\
```sfreq``` is hardcoded in our example, however, feel free to change ```τ``` with the slider below.
"""

# ╔═╡ ec2dd388-3b46-4db8-982d-65e9dd53971f
begin
	window_slider = md"""Change window size τ = (-1.0, $(@bind τ  Slider(-1:0.1:10,default=7,show_value=true)))"""
end

# ╔═╡ 3a463dec-f3e5-4bb0-a887-4e06625a4dd5
md"""
!!! tip \"Epoching in Matlab\" 
	In Matlab you can either use EEGLABs native epoching function or uf_epoch() for this.
"""

# ╔═╡ baa405cb-b3f1-4afa-8bd0-8945f861186e
md"""
!!! warning \"Question\" 
	Have a look at the size of data_epochs. The dimensions give you channel X sample X epoch. What seems off here? What would you expect in a real EEG?

	We simulated the data this way purely to make it computationally less heavy.
"""

# ╔═╡ 2362e18f-0240-4a88-b241-4aa278290864
md"""
## Model Set-Up and Fit
"""

# ╔═╡ 3b5afbc8-2658-4d76-849e-554b2663f38b
md"""
Now that we have our data in the right format we can finally set up a Mass Univariate model to get our Kernels back!\
For this, we first need to define a formula for our model (in Wilkinson notation).
In Julia this is rather trivial with the StatsModels package:

```julia
	formula  = @formula 0 ~ 0 + conditionA + conditionB;
```


!!! note \"OBS!!!\" 
	Note that we are omitting an intercept term. This is because of the way we set up our events table. We could also have coded the table in a different way; for more on this, I recommend you delve deeper into dummy/ contrast coding of regression models.

And finally, we can fit our model:
```julia
fit(UnfoldModel,formula,events,data_epochs,times)
```


However, the model results can seem a bit confusing, so let's quickly bring this into a better format using: ```coeftable(m) ```, where m is the output from unfolds ```fit()``` function. 

You can see below how this looks in practice.
"""

# ╔═╡ 16383a33-9dbd-415e-8fc8-61b6ed05e433
md"""
!!! tip \"Fitting the model\" 
	Again, this works slightly different in Matlab, the cheat-sheet can help you out here as well.
"""

# ╔═╡ 5957d8db-aaf1-4210-ba3f-563d7fd466f2
md"""
## Results
"""

# ╔═╡ 6ba79973-63ce-416e-a223-f4420e7d1506
md"""
Let's vizualize our results:
"""

# ╔═╡ aa7b5e40-8a29-4e9a-9fd4-b584f027ffc4
md"""
!!! tip \"Plotting results\" 
	We will be using Julias Plots.jl package for this to give you a better understanding of the results table. However, both the Matlab and Julia versions of Unfold have their own plotting functions (for Julia this is incorporated in the UnfoldMakie.jl package)
"""

# ╔═╡ 86855f65-4735-4dfd-a4ab-f4c6244d79fb
md"""
Looks like everything worked and we got our kernels back!
"""

# ╔═╡ ecd9e8ee-780e-4e7e-bfbd-724de8a57293
md"""
!!! warning \"Question\" 
	All we did for now was to qualitatively show that the method works, how would we investigate this more systematically?
"""

# ╔═╡ 1af20e94-4eaa-4a2c-886d-b556e3a36f7d
begin
	# condition A
	#condA_mass = filter(row->row.coefname=="(Intercept)", results_massU)
	#plot(condA_mass.time, condA_mass.estimate, ylims=(-5,5), linecolor=:orange, 
		#label="conditionA", legend=:outerbottom)

	# condition B
	#condB_mass = filter(row->row.coefname=="condition", results_massU)
	#plot!(condB_mass.time, condB_mass.estimate .+ condA_mass.estimate, ylims=(-5,5), linecolor=:deepskyblue, 
		#label="conditionB")

	#vline!([0], linestyle=:dash, linecolor=:black, label="")
end

# ╔═╡ 749a6ca6-2322-4df9-9355-d3f7b42cc760
md"""
## What is Overlap?
"""

# ╔═╡ b7ac9c1e-aabc-45c3-80b5-c85e5bd233ff
md"""
!!! note \"OBS!!\" 

	So far we made an assumption about ERPs which is not met by most EEG experiments (and certainly not "in the wild"). Namely, we assumed that signals do not overlap with each other.
	
	PS: There are more assumptions in ERP research that are unlikely true, however, let's focus on the overlap for now.

"""

# ╔═╡ 97ef0471-39aa-41fb-b3dc-751aeb0751e2
md"""
To better understand what happens to the signal if different response kernels/ ERPs overlap let's change our simulation to do exactly that.
"""

# ╔═╡ a794dca3-0fda-4649-9517-da34b9023761
begin
	overlap = md"""Choose if Overlap should be simulated: $(
		@bind overlap₁ Select(
			[1=>"no Overlap", 2=> "Overlap"],
		default=1))"""
end

# ╔═╡ 8472b1ed-b784-4aea-98ee-641f3d827401
md"""
Additionally, you can change the amount of overlap with the two sliders below. But be careful, depending on the settings you might run into memory errors.
"""

# ╔═╡ add0a511-6ed0-4d74-9486-debb5cfc54d7
begin	
	noise1_slider = md"""Change noise: σ\_1 = $(@bind σ₁ Slider(1.:0.01:6, default=1.0, show_value=true))"""
end

# ╔═╡ 53bbb4a7-9052-417a-b8f3-3230608eb85c
begin	
	mean_slider = md"""Change mean:μ\_1 = $(@bind μ Slider(0:0.01:5, default=0, 
		show_value=true))"""
end

# ╔═╡ 31462217-1b12-4ea1-aea8-323045b2167e
begin
	style = "
		position: fixed;
		right: 1rem;
		top: 1rem;
		width: 25%;
		padding: 10px;
		border: 3px solid rgba(0, 0, 0, 0.15);
		border-radius: 10px;
		box-shadow: 0 0 11px 0px #00000010;
		max-height: calc(100vh - 5rem - 56px);
		overflow: auto;
		z-index: 10;
		background: white;
		color: grey;
	";
	sidebar = Div([
		html"""<nav class="plutoui-toc">
			<header style="color: hsl(0,0%,25%) !important">
			Interactive Sliders
			</header>
			</nav>""",
		md"""Here are all interactive bits of the notebook at one place.\
		Feel free to change them!""",
		md"""-----""",
		function_A_selection,
		l_A,
		md"""-----""",
		function_B_selection,
		l_B,
		md"""---""",
		noise1_slider,
		mean_slider,
		md"""---""",
		noise_slider,
		window_slider,
		overlap,
	], style=style)
end

# ╔═╡ 12fceea6-7dfe-4e02-886a-eb52fe326529
begin
	if overlap₁ == 2
		# sample event onsets
		event_onsets_A = sort(sample(MersenneTwister(8),1:6000, 300, replace = false))
		event_onsets_B = event_onsets_A + rand(LogNormal(μ, σ₁),300)
		#event_onsets_B = sort(sample(MersenneTwister(1),1:6000, 300, replace = false))
		[event_onsets_A event_onsets_B]' # for display

		# graph of event onsets for stimuli A
		e1 = vline(event_onsets_A, xlims=(0,100), ylims=(0,1), 	
			linecolor=:orange,linestyle=:dash, label="event onset of stimuli A")

		# graph of event onsets for stimuli B
		e2 = vline!(event_onsets_B, xlims=(0,100),ylims=(0,1), 
			linecolor=:deepskyblue,linestyle=:dash, label="event onset of stimuli B")

	elseif overlap₁ == 1
		
		event_onsets_AB =  [3 15]
		for i in 3:2:600 event_onsets_AB = hcat(event_onsets_AB, (event_onsets_AB[[i-2, i-1]]'.+20)) end

		event_onsets_A = event_onsets_AB[1:2:lastindex(event_onsets_AB)]
		event_onsets_B = event_onsets_AB[2:2:lastindex(event_onsets_AB)]
			
		# graph of event onsets for stimuli A
		e1 = vline(event_onsets_A, xlims=(0,100), ylims=(0,1), 	
			linecolor=:orange,linestyle=:dash, label="event onset of stimuli A")
		
		e2 = vline!(event_onsets_B, xlims=(0,100),ylims=(0,1), 
			linecolor=:deepskyblue,linestyle=:dash, label="event onsets of stimuli B")
	end

	# plotting
	plot(e2, size=(600,200))
	
end

# ╔═╡ 084f9199-5625-42d8-ac80-f0b75575569b
begin
	# assemble the separate signals via the event onsets
	eeg_A(t) = sum((0, (erp_A(t-a) for a in event_onsets_A if abs(t-a)<10)...))
	eeg_B(t) = sum((0, (erp_B(t-a) for a in event_onsets_B if abs(t-a)<10)...))
	
	# addition of the separate signals at each timepoint
	eeg(t) = eeg_A(t) .+ eeg_B(t)
	#eegT = add_gauss(eeg, NoiseL)
end;

# ╔═╡ 85f7f440-7bd2-45cf-abb9-99501aa3b5b5
begin
	range = 0:0.1:6000;
	data = eeg.(range);
	data_noise = data .+ σ .* randn(size(data));
end;

# ╔═╡ fac2c2ba-cedf-4291-bb0a-1951cb173a8e
let
	p1 = plot(range[1:801], data_noise[1:801], xlims=(-10, 70), ylims=(-5,5), legend=false,  linecolor=:green, size=(600, 200))
	vline!([0], linestyle=:dash, linecolor=:black)
end

# ╔═╡ c568b352-374b-48ea-9cd3-99f7c1dbb88d
begin
	p1 = plot(-10:0.1:70, eeg, xlims=(-10, 70), ylims=(-5,5), legend=false,  linecolor=:green)
	vline!([0], linestyle=:dash, linecolor=:black)
	p2 = plot(-10:0.1:70,eeg_A, xlims=(-10, 70), ylims=(-5,5), legend=false, linecolor=:orange)
	plot!(-10:0.1:70,eeg_B, xlims=(-10, 70), ylims=(-5,5), legend=false, linecolor=:deepskyblue)
	vline!([0], linestyle=:dash, linecolor=:black)
	vline!(event_onsets_A,linecolor=:orange,linestyle=:dash)
	vline!(event_onsets_B,linecolor=:deepskyblue,linestyle=:dash)
	plot(p2, p1,layout=@layout[a;b])
	
end

# ╔═╡ 3dc142c1-27c9-4a6a-8c01-5a39a2fb7f90
begin
	# convert the created function into a event dataframe for unfold
	dummy = false
	if dummy
		events_A = DataFrame(latency = event_onsets_A, condition=0);
		events_B = DataFrame(latency = event_onsets_B, condition=1);
		events = sort(vcat(events_A, events_B), [:latency])
	else
		events_A = DataFrame(latency = event_onsets_A, conditionA=1);
		events_B = DataFrame(latency = event_onsets_B, conditionB=1);
		events = sort(outerjoin(events_A, events_B, on = :latency), [:latency])
	end
	
	insertcols!(events, 2, :intercept => 1)
	insertcols!(events, 2, :type => "stimulus")
	events = coalesce.(events, 0);
	events.latency = events.latency * 10
end;

# ╔═╡ d6797645-e1d0-4449-888a-c8c348a5cef5
events[1:5,:]

# ╔═╡ e2895365-393e-46a9-a5ae-9c361b51f845
begin
	# cut the data into epochs
	data_epochs,times = Unfold.epoch(data=data_noise,tbl=events,τ=(-1.0,τ ),sfreq=10);
end;

# ╔═╡ 04283223-19d7-4e8e-b6e5-4e554fc7c736
# Show the size of epoched data
size(data_epochs)

# ╔═╡ 9e117fac-6ca4-4432-a199-9d8777eec277
begin
	# formula in wilikinson notation
	formula  = @formula 0 ~ 0 + conditionA + conditionB;
	#formula  = @formula 0 ~ 1 + condition;

	# And as easy as that we can fit our model
	m_massU = fit(UnfoldModel,formula,events,data_epochs,times);

	# Let's bring our results into a tidy format
	results_massU = coeftable(m_massU)
end

# ╔═╡ 087e6dcb-ab3a-41fb-ba31-43909b30e6d7
begin
	# condition A
	condA_massU = filter(row->row.coefname=="conditionA", results_massU)
	plot(condA_massU.time, condA_massU.estimate, ylims=(-5,5), linecolor=:orange, 
		label="conditionA", legend=:outerbottom)

	# condition B
	condB_massU = filter(row->row.coefname=="conditionB", results_massU)
	pₒ = plot!(condB_massU.time, condB_massU.estimate, ylims=(-5,7), linecolor=:deepskyblue, 
		label="conditionB")

	vline!([0], linestyle=:dash, linecolor=:black, label="")
end

# ╔═╡ d72e80fe-aa3f-474b-b007-8e20a484d6a2
md"""
---
As before, we can visualize our event onsets as well as the resulting signals to get an idea of our data.
"""

# ╔═╡ ba2ae4c5-b1c8-4065-8235-36851add9975
plot(e2, size=(600,200))

# ╔═╡ dd818514-b764-4651-a0e9-3b15c2b5748d
plot(p2, p1,layout=@layout[a;b])

# ╔═╡ fe92f817-2c77-462b-8235-74c1462c297d
begin
	plot(range[1:801], data_noise[1:801], xlims=(-10, 70), ylims=(-5,5), legend=false,  linecolor=:green, size=(600, 200))
	vline!([0], linestyle=:dash, linecolor=:black)
end
	

# ╔═╡ 44a58bbd-4ec8-416a-9a4a-b0f39f96d3e8
md"""
---
And now let's have a look again at the results from our Mass-Univariate Model.
"""

# ╔═╡ f992532e-7360-47f7-810f-802a148f6571
begin
		# condition A
	plot(condA_massU.time, condA_massU.estimate, ylims=(-5,5), linecolor=:orange, 
		label="conditionA", legend=:outerbottom)

	# condition B
	plot!(condB_massU.time, condB_massU.estimate, ylims=(-5,7), linecolor=:deepskyblue, 
		label="conditionB")

	vline!([0], linestyle=:dash, linecolor=:black, label="")
end

# ╔═╡ aa6465bc-4c2c-4a4e-877d-cd54099c6135
md"""
!!! note \"OBS!!\" 

	This doesn't look like our original response kernels anymore!!!
"""

# ╔═╡ 5fa9df7b-557b-4ba8-af4f-d39f7b6135dc
md"""
Now, what is happening here? For this, we first need to understand that every recorded EEG signal is a convolution of a multitude of signals. Usually, additional signals convolve with our signal of interest at random (i.e. seldom repeatedly) and we can classify this as noise. And simply by averaging over a multitude of trials, we can get rid of such noise (which is why the "traditional" approach became so popular).\

However, as soon as two signals are convolved repeatedly, either because Stimuli are presented too close together or because one event is dependent on another (e.g. a participants response), simply averaging doesn't help us get rid of the influence the signals have on each other.\

In the following, you will now see why it makes sense to transition from simple averaging to a regression framework (and what powerful tools this gives us).
"""

# ╔═╡ ca9a9017-33ed-4b05-8140-ca8350698c95
md"""
!!! warning \"Question\" 
	Can you think of specific experimental setups where overlap might pose a problem? 
"""

# ╔═╡ 35dfc2e1-1653-498a-9b13-ec7038c82973
md"""
## Overlap correction
"""

# ╔═╡ 993c86b0-d7cc-41ce-b60b-1e226b270c18
md"""
To disentangle two convolved signals from each other we can use the inverse transformation, a deconvolution. And luckily for us, it is (more or less) easy to incorporate this into a regression model. The only thing we have to do is to provide the model with the continuous data and give it a set of basis functions, which it can use to "learn" how the different signals influence each other.\

If you want to go more in-depth about how this works under the hood, I would point you to the [original Unfold paper](https://peerj.com/articles/7838/#p-28) for now.
"""

# ╔═╡ e2cb915c-a3a5-43ca-a807-54b6bd02f658
md"""
!!! note \"OBS!!\" 
	Deconvolution only works when there is some variation in the overlap of your signals!\
	If, for example, due to your experimental set-up one stimulus is always exactly 800ms after the first the model won't be able to learn anything about the overlap from your data.
"""

# ╔═╡ 0e006a95-d7c1-4149-9f2d-d14a91d138ac
md"""
In practice, Unfold does all the work for us. And the only thing we have to change compared to our Mass-Univariate model from before is that we have to give Unfold the continuous data (instead of the epoched one) and define a set of basis functions alongside our model formula.
"""

# ╔═╡ 238faddf-1f2a-4a10-991e-afc3f85e4790
begin
	# Formula
	f = @formula 0~0+conditionA+conditionB

	# basisfunction via FIR basis
	basisfunction = firbasis(τ=(-1.0,τ), sfreq=10, name="stimulus")
	
	# map basidfunction & formula into a dict
	bfDict = Dict(Any=>(f,basisfunction))

	# fit model
	m = fit(UnfoldModel,bfDict,events, data_noise);

	# create result dataframe
	results = coeftable(m);
end

# ╔═╡ 7648bb02-af27-4220-bb00-392696bbd0c1
md"""
---
As before, we are getting a tidy results table which we can use to plot our results and compare them with the non-overlap corrected approach.
"""

# ╔═╡ 2a50c197-70dd-4fdc-957c-0e50718d0197
begin
	# condition A
	condA = filter(row->row.coefname=="conditionA", results)
	plot(condA.time, [condA.estimate condA_massU.estimate], layout = (2,1), ylims=(-5,5), linecolor=:orange, 
		label="conditionA", legend=:outerbottom, title=["Overlap Corrected" "No Overlap Correction"])

	# condition B
	condB = filter(row->row.coefname=="conditionB", results)
	plot!(condB.time, [condB.estimate condB_massU.estimate], ylims=(-5,5), linecolor=:deepskyblue, 
		label="conditionB")
	plot!(size=(700,800))

	vline!([0], linestyle=:dash, linecolor=:black, label="")
end

# ╔═╡ 2152252e-2b34-4338-aed3-f041ce75b599
md"""
And there we have it, we got our original signal kernels back despite a confounding overlap!
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsModels = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Unfold = "181c99d8-e21b-4ff3-b70b-c233eddec679"

[compat]
DSP = "~0.7.5"
DataFrames = "~1.3.2"
Distributions = "~0.25.50"
Plots = "~1.27.0"
PlutoUI = "~0.7.37"
StatsBase = "~0.33.16"
StatsModels = "~0.6.29"
StatsPlots = "~0.14.33"
Unfold = "~0.3.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Arrow]]
deps = ["ArrowTypes", "BitIntegers", "CodecLz4", "CodecZstd", "DataAPI", "Dates", "Mmap", "PooledArrays", "SentinelArrays", "Tables", "TimeZones", "UUIDs"]
git-tree-sha1 = "85013d248b128cf13ae62c827c4bf05872e97f78"
uuid = "69666777-d1a9-59fb-9406-91d4454c9d45"
version = "2.2.1"

[[deps.ArrowTypes]]
deps = ["UUIDs"]
git-tree-sha1 = "a0633b6d6efabf3f76dacd6eb1b3ec6c42ab0552"
uuid = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
version = "1.2.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BSplines]]
deps = ["LinearAlgebra", "OffsetArrays", "RecipesBase"]
git-tree-sha1 = "5b609325fcb8f5fc124351b9267183722965860d"
uuid = "488c2830-172b-11e9-1591-253b8a7df96d"
version = "0.3.3"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "5a814467bda636f3dde5c4ef83c30dd0a19928e0"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.2.6"

[[deps.BlockDiagonals]]
deps = ["ChainRulesCore", "FillArrays", "FiniteDifferences", "LinearAlgebra"]
git-tree-sha1 = "e256e3aefd8041524f7338f655caa42329a31f5b"
uuid = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
version = "0.1.26"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "5196120341b6dfe3ee5f33cf97392a05d6fe80d0"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.4"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c9a6160317d1abe9c44b3beb367fd448117679ca"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.13.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "59fe0cb37784288d6b9f1baebddbf75457395d40"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.CodecZstd]]
deps = ["CEnum", "TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "849470b337d0fa8449c21061de922386f32949d9"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.7.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "3e03979d16275ed5d9078d50327332c546e24e68"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.5"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "ae02104e835f219b8930c7664b8012c93475c340"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "a49fef6d584d1f585baebe9713a6d43af3db5fc8"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.50"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "90b158083179a6ccbce2c7eb1446d5bf9d7ae571"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.7"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Effects]]
deps = ["DataFrames", "LinearAlgebra", "Statistics", "StatsBase", "StatsModels", "Tables"]
git-tree-sha1 = "f99ed3dd68cf67f9b3c78ea30a7ab15a527eafc7"
uuid = "8f03c58b-bd97-4933-a826-f71b64d2cca2"
version = "0.1.5"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "505876577b5481e50d089c1c68899dfb6faebc62"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.6"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "0dbc5b9683245f905993b51d2814202d75b34f1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.1"

[[deps.FiniteDifferences]]
deps = ["ChainRulesCore", "LinearAlgebra", "Printf", "Random", "Richardson", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "0ee1275eb003b6fc7325cb14301665d1072abda1"
uuid = "26cc04aa-876d-5657-8c51-4c34ba976000"
version = "0.12.24"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "fb764dacfa30f948d52a6a4269ae293a479bbc62"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.6.1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9f836fb62492f4b0f0d3b06f55983f2704ed0883"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a6c850d77ad5118ad3be4bd188919ce97fffac47"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "SpecialFunctions", "Test"]
git-tree-sha1 = "65e4589030ef3c44d3b90bdc5aac462b4bb05567"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.8"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IncompleteLU]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a22b92ffedeb499383720dfedcd473deb9608b62"
uuid = "40713840-3770-5561-ab4c-a76e7d0d7895"
version = "0.2.0"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b15fc0a95c564ca2e0a7ae12c1f095ca848ceb31"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.5"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "8c1f668b24d999fb47baf80436194fdccec65ad2"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "4f00cc36fede3c04b8acf9b2e2763decfdcecfa6"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.13"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase"]
git-tree-sha1 = "3bd9fd4baf19dfc1edf344bc578da7f565da2e18"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.9.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "a62df301482a41cb7b1db095a4e6949ba7eb3349"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.1.0"

[[deps.MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.MixedModels]]
deps = ["Arrow", "DataAPI", "Distributions", "GLM", "JSON3", "LazyArtifacts", "LinearAlgebra", "Markdown", "NLopt", "PooledArrays", "ProgressMeter", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StatsModels", "StructTypes", "Tables"]
git-tree-sha1 = "64bd164a79a4a27ef56db494e14e0cf6e76c3658"
uuid = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
version = "4.6.1"

[[deps.MixedModelsPermutations]]
deps = ["BlockDiagonals", "LinearAlgebra", "MixedModels", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsModels", "Tables"]
git-tree-sha1 = "a08c290a3b4770866c25c732aad6b38d196a041e"
uuid = "647c4018-d7ef-4d03-a0cc-8889a722319e"
version = "0.1.4"

[[deps.MixedModelsSim]]
deps = ["LinearAlgebra", "MixedModels", "PooledArrays", "PrettyTables", "Random", "Statistics", "Tables"]
git-tree-sha1 = "96ce9a3dd9499fd679a4ffd494d339d50248da0e"
uuid = "d5ae56c5-23ca-4a1f-b505-9fc4796fc1fe"
version = "0.2.6"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "29714d0a7a8083bba8427a4fbfb00a540c681ce7"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "ba8c0f8732a24facba709388c74ba99dcbfdda1e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.0"

[[deps.NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "5a7e32c569200a8a03c3d55d286254b0321cd262"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.5"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9b1f15a08f9d00cdb2761dcfa6f453f5d0d6f973"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.1+0"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "648107615c15d4e09f7eca16307bc821c1f718d8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e8185b83b9fc56eb6456200e873ce598ebc7f262"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.7"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgBenchmark]]
deps = ["BenchmarkTools", "Dates", "InteractiveUtils", "JSON", "LibGit2", "Logging", "Pkg", "Printf", "TerminalLoggers", "UUIDs"]
git-tree-sha1 = "e4a10b7cdb7ec836850e43a4cee196f4e7b02756"
uuid = "32113eaa-f34f-5b0d-bd6c-c81e245fc73d"
version = "0.2.12"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "9213b4c18b57b7020ee20f33a4ba49eb7bef85e0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "0107e2f7f90cc7f756fee8a304987c574bbd7583"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.0.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "1fc929f47d7c151c839c5fc1375929766fb8edcc"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.1"

[[deps.PyMNE]]
deps = ["PyCall"]
git-tree-sha1 = "b3caa6ea95490974465487d54fc1e62a094bad8e"
uuid = "6c5003b2-cbe8-491c-a0d1-70088e6a0fd6"
version = "0.1.2"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "995a812c6f7edea7527bb570f0ac39d0fb15663c"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Richardson]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e03ca566bec93f8a3aeb059c8ef102f268a38949"
uuid = "708f8203-808e-40c0-ba2d-98a6953ed40d"
version = "1.4.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "6a2f7d70512d205ca8c7ee31bfa9f142fe74310c"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.12"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "74fb527333e72ada2dd9ef77d98e4991fb185f04"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "25405d7016a47cf2bd6cd91e66f4de437fd54a07"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.16"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "03c99c7ef267c8526953cafe3c4239656693b8ab"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.29"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "4d9c69d65f1b270ad092de0abe13e859b8c55cad"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.33"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimeZones]]
deps = ["Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "2d4b6de8676b34525ac518de36006dc2e89c7e2e"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.7.2"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "d60b0c96a16aaa42138d5d38ad386df672cb8bd8"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.16"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unfold]]
deps = ["BSplines", "CategoricalArrays", "DSP", "DataFrames", "Distributions", "DocStringExtensions", "Effects", "GLM", "IncompleteLU", "IterativeSolvers", "LinearAlgebra", "Logging", "MLBase", "Missings", "MixedModels", "MixedModelsPermutations", "MixedModelsSim", "PkgBenchmark", "ProgressMeter", "PyMNE", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StatsModels", "Tables", "Test", "TimerOutputs"]
git-tree-sha1 = "16f57c4b29637c358059bf171cd1873a80b6a5cb"
uuid = "181c99d8-e21b-4ff3-b70b-c233eddec679"
version = "0.3.8"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "505c31f585405fc375d99d02588f6ceaba791241"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.5"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─3e17c9f0-a453-11ec-2206-0bfea70f79d3
# ╟─ccd268ce-dfed-4dd2-8cc2-1d9a39c0e67d
# ╟─18bfbf75-5e8d-4bbf-880f-589cbac1999e
# ╟─09beb045-a5d2-4472-8543-a95e6136e2a0
# ╟─1c69a368-8fe7-4fe2-8a94-562f06f38bfd
# ╟─f0d58bb0-f673-448d-851d-0adb718d5cfe
# ╟─9697844d-3da2-4b3f-811b-eeb075b90fdd
# ╟─32303686-90db-4abb-ba87-57a58f93af1d
# ╠═81ca9393-5d98-4d85-96fc-4725787aed18
# ╟─31462217-1b12-4ea1-aea8-323045b2167e
# ╟─d2633e8f-dd86-453b-ac83-61c8990f2723
# ╟─c84371c6-9bfb-4632-b641-43612d41023b
# ╟─4c48a3ac-2fd2-4462-a655-559e69c4a18b
# ╟─1ae08d84-de51-439c-a580-4e65c45a1dc8
# ╟─bf17e5d8-afcf-4537-a25f-0bc3c7389887
# ╟─61928269-a3a7-41b6-beb3-0b5acfac6299
# ╟─bfc4c110-4e16-4473-b898-0d0f6d4fd21f
# ╟─7a6ed8c9-c9cc-4651-8475-ce3e25bce26c
# ╟─658ae8ea-061d-4fa2-af6a-690459b40849
# ╟─f8b9ab1a-ff14-4e6e-aa8d-e1fb2a49191a
# ╟─d3f9adc8-9e3f-4d68-bf1a-5fc526a0cd40
# ╟─339ecc3a-d5cf-452d-b923-1e49e01839ed
# ╟─b05ca432-e0d6-4790-9dfd-ab303ec9abc7
# ╟─38e57913-991c-4afe-8895-3901eceb9783
# ╟─74315375-2094-4779-826a-30b7b980d659
# ╟─7de9ee9d-6176-4923-9667-ded47146862e
# ╟─7e1eb3ae-829b-419b-8d7f-b08174419b19
# ╟─12fceea6-7dfe-4e02-886a-eb52fe326529
# ╟─084f9199-5625-42d8-ac80-f0b75575569b
# ╟─81b7896b-054d-4827-b9b5-a983f39ad674
# ╟─c568b352-374b-48ea-9cd3-99f7c1dbb88d
# ╟─7aeeb225-00eb-4197-b360-8d44e6033375
# ╟─d4b54556-f14b-4f31-bba3-e71b077d32ee
# ╟─b58dadff-334c-46e8-9135-891d5d9f33d2
# ╟─7e42a173-4efd-4d0f-8044-251237416514
# ╟─85f7f440-7bd2-45cf-abb9-99501aa3b5b5
# ╟─fac2c2ba-cedf-4291-bb0a-1951cb173a8e
# ╟─e086d4c4-a732-4f7d-a430-905190f3b12e
# ╟─56c9a93f-4b2a-4891-9a1c-21e57b557911
# ╟─4f973b6e-f740-422f-a66b-98a46cc97790
# ╟─3dc142c1-27c9-4a6a-8c01-5a39a2fb7f90
# ╟─d6797645-e1d0-4449-888a-c8c348a5cef5
# ╟─fdc2a51f-5c75-43ef-9384-0b6521072548
# ╟─47f80bf2-7c90-4aeb-ac4c-1db2a812285f
# ╟─ec2dd388-3b46-4db8-982d-65e9dd53971f
# ╠═e2895365-393e-46a9-a5ae-9c361b51f845
# ╟─3a463dec-f3e5-4bb0-a887-4e06625a4dd5
# ╠═04283223-19d7-4e8e-b6e5-4e554fc7c736
# ╟─baa405cb-b3f1-4afa-8bd0-8945f861186e
# ╟─2362e18f-0240-4a88-b241-4aa278290864
# ╟─3b5afbc8-2658-4d76-849e-554b2663f38b
# ╠═9e117fac-6ca4-4432-a199-9d8777eec277
# ╟─16383a33-9dbd-415e-8fc8-61b6ed05e433
# ╟─5957d8db-aaf1-4210-ba3f-563d7fd466f2
# ╟─6ba79973-63ce-416e-a223-f4420e7d1506
# ╟─aa7b5e40-8a29-4e9a-9fd4-b584f027ffc4
# ╠═087e6dcb-ab3a-41fb-ba31-43909b30e6d7
# ╟─86855f65-4735-4dfd-a4ab-f4c6244d79fb
# ╟─ecd9e8ee-780e-4e7e-bfbd-724de8a57293
# ╟─1af20e94-4eaa-4a2c-886d-b556e3a36f7d
# ╟─749a6ca6-2322-4df9-9355-d3f7b42cc760
# ╟─b7ac9c1e-aabc-45c3-80b5-c85e5bd233ff
# ╟─97ef0471-39aa-41fb-b3dc-751aeb0751e2
# ╟─a794dca3-0fda-4649-9517-da34b9023761
# ╟─8472b1ed-b784-4aea-98ee-641f3d827401
# ╟─add0a511-6ed0-4d74-9486-debb5cfc54d7
# ╟─53bbb4a7-9052-417a-b8f3-3230608eb85c
# ╟─d72e80fe-aa3f-474b-b007-8e20a484d6a2
# ╟─ba2ae4c5-b1c8-4065-8235-36851add9975
# ╟─dd818514-b764-4651-a0e9-3b15c2b5748d
# ╟─fe92f817-2c77-462b-8235-74c1462c297d
# ╟─44a58bbd-4ec8-416a-9a4a-b0f39f96d3e8
# ╠═f992532e-7360-47f7-810f-802a148f6571
# ╟─aa6465bc-4c2c-4a4e-877d-cd54099c6135
# ╟─5fa9df7b-557b-4ba8-af4f-d39f7b6135dc
# ╟─ca9a9017-33ed-4b05-8140-ca8350698c95
# ╟─35dfc2e1-1653-498a-9b13-ec7038c82973
# ╟─993c86b0-d7cc-41ce-b60b-1e226b270c18
# ╟─e2cb915c-a3a5-43ca-a807-54b6bd02f658
# ╟─0e006a95-d7c1-4149-9f2d-d14a91d138ac
# ╠═238faddf-1f2a-4a10-991e-afc3f85e4790
# ╟─7648bb02-af27-4220-bb00-392696bbd0c1
# ╟─2a50c197-70dd-4fdc-957c-0e50718d0197
# ╟─2152252e-2b34-4338-aed3-f041ce75b599
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
