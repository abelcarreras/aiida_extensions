

def _generate_LAMMPS_potential(data):
    return None

def _get_input_potential_lines(data, names=None, potential_filename='potential.pot'):
    lammps_input_text = 'pair_style  lj/cut 2.5\n'
    for line in data:
        lammps_input_text += 'pair_coeff  {}\n'.format(' '.join(line))
    return lammps_input_text
