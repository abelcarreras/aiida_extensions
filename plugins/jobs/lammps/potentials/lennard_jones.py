

def _generate_LAMMPS_potential(data):
    return None

def _get_input_potential_lines(data, names=None, potential_filename='potential.pot'):
    cut = 5.0
    lammps_input_text = 'pair_style  lj/cut {}\n'.format(cut)

    for key, value in data.iteritems():
        lammps_input_text += 'pair_coeff {}    {}\n'.format(key, value)
    return lammps_input_text
