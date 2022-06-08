# Usage
There are two ways to using the Pluto Notebooks in this repository.
The first one (using Binder) is easy, doesn't require you to set up anything and runs online. However, this method sometimes (often) doesn't work, is slow and some functionality might be a bit off.

If this doesn't work for you, you can still install Julia, add Pluto.jl and run everything locally.

## Using Julia and Pluto.jl (recommended)
First you need to either download the specific notebook file you want to use (from the src folder) or clone/ download the entire repo.

If Julia is not already installed on your computer go to https://julialang.org/downloads/ and download your respective version. I used Julia Version 1.7.1, so every version past that should work as well. Now just start and follow the wizard.

Next you want to open any command prompt (for Windows tap the windows key and type "cmd") and type "julia" into the command line. This should've started julia. Now type "]" to get to the package manager (the green "julia" text should be replaced by a blue version number) and type "add Pluto".

After Pluto is installed hit backslash to exit the package manager and execute the commands "using Pluto" and subsequently "Pluto.run()" to open a Pluto session in your default browser. From here you can paste the (absolut local) path to your desired notebook and start exploring.

## Binder
Simply go to http://pluto-on-binder.glitch.me/ and copy the permalink from any Pluto notebook (e.g. https://github.com/ReneSkukies/PlutoNotebooks/blob/main/src/nb_unfold_introduction.jl ) into the text field and click the binder link.
