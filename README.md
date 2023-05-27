# Parallel spectra calculator
<a name="readme-top"></a>
<p align="center">
  <img width="1300" height="200" src="https://github.com/kaifengZheng/FEFF_package/assets/48105165/ccccd72e-1292-4875-8232-dc87a0f0967e">
</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#Packages">Packages</a></li>
        <li><a href="#Usage">Usage</a></li>
      </ul>
    </li>
    <li>
      <a href="#Workflow">Workflow</a>
      <ul>
        <li><a href="#writing process">Prerequisites</a></li>
        <li><a href="#running process">Installation</a></li>
      </ul>
    </li>
    <li><a href="#examples">examples</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
<p align="right">(<a href="#readme-top">back to top</a>)</p>
## About The Project
This package can be used to run multiple FEFF calculations in parallel, decreasing the amount of calculation time when one wants to run huge FEFF calculations. For instance, one may be interested in particle-averaged spectra for different shape particles. If those particles are irregular, most likely, every atom is inequivalent. Therefore, we need to calculate thousands or hundred thousand spectra. Using either the sequential or mpi version of FEFF is impossible to run sequentially and finish the jobs in a reasonable time(3000 147-atom particles require 300 more days to finish it！Using this package will decrease the calculation time dramatically and will finish them within one week(use 3x96 core on cluster). <br>

Why I design this package:
* Decrease the calculation time ⏲️
* Make the FEFF calculation easier ⚛️
* Provide useful functions(average) 😸

The functionality of the package:
* run calculations(now only for FEFF) in parallel
* can restart the calculations(now only for FEFF) from the checkpoint
* allow coordinates as the input file
* can use SCF to tune calculation(now only for FEFF) parameters
* flexable modes for running the calculation
* Only support linux system(parallel mode needs mpi support)
* `toolbox` provides the average method

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Packages
This section should list any major libraries and softwares used to bootstrap my project. 
<br><ins>Spectra calculators:<ins>
* FEFF https://feff.phys.washington.edu/<br>

<ins>Python packages:</ins><br>
* numpy https://numpy.org/
* pandas https://pandas.pydata.org/
* Pymatgen https://pymatgen.org/
* ASE https://wiki.fysik.dtu.dk/ase/
* mpi4py https://mpi4py.readthedocs.io/en/stable/
* mpipool https://pypi.org/project/mpipool/

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Environment setting
First, configure the correct mpi environment on your computer. You could check my configurations in the `run.slurm`. open `module_file/FEFF/10.0.0`, put your FEFF package path and mpi configurations in the following lines:
*environment configuration
 ```sh
  set               root              [add FEFF dir here!]


if { ![ is-loaded intel/oneAPI/2022.2 ] }         { module load intel/oneAPI/2022.2 }
if { ![ is-loaded mpi ] }         { module load mpi }
if { ![ is-loaded mkl ] }         { module load mkl }
```
Put the `FEFF_files/feff` and `FEFF_files/feffmpi` in the place you like, and change the `PATH` inside those files to the `feff/bin` path.<br>
  
<br>make sure FEFF package is built on your system, and then go into `module_file`, and run:
* module
  ```sh
  module use FEFF
  module load 10.0.0
  ```

  
 It will enable the FEFF10 calculator, and then you can run FEFF simply by click:
 * run FEFF
 ```sh
 feff #for sequential FEFF
 feffmpi 8 #for mpi feff running on 8 cpus
 ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration file
The configurations of the calculation stores in toml format. Please check [Documentation]()
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Run
Create a new diractory and clone the whole package into it. The main program is `FEFF_run_v3.py`. This code will treat writing and running as sperate processes.<br>
For writing process, one needs to prepare `input` directory to store all coordinates files(xyz,cif,POSCAR),confiugration file: `config.toml`. and template file:`template.inp`. Running by use:
  ```sh
  mpirun -np [number of cores] python -m mpi4py FEFF_run_v3.py -w
  ```
For running process, one needs `config.toml`, 'FEFF_inp' directory(which generated by writing process),`template.inp`. Running by use:
  ```sh
  mpirun -np [number of cores] python -m mpi4py FEFF_run_v3.py -r
  ``` 

<!-- WORKFLOW -->
 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Workflow

<p align="center">
  <img width="1300" height="800" src="https://github.com/kaifengZheng/FEFF_package/assets/48105165/cc5cda6e-a2bb-4c6d-928e-06a6859f948c">
</p>

### Prerequisites

People can install all the dependency using:
  ```sh
  pip install -r requirements.txt
  ```

<!-- USAGE EXAMPLES -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Examples

Some cif examples are stored in `cif_input_samples`, and xyz examples are stored in `input`.<br>

_For more examples, please refer to the [Documentation]()




<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Kaifeng Zheng - kaifeng.zheng@stonybrook.edu

Project Link: [https://github.com/kaifengZheng/FEFF_package.git](https://github.com/kaifengZheng/FEFF_package.git)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 


