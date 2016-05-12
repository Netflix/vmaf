# dictionary of known host/profile pairs
"""
high-level programming interface to pathos host registry
"""

default_profile = '.bash_profile'
_profiles = { }
"""
For example, to register two 'known' host profiles:

  _profiles = { \
     'foobar.danse.us':'.profile',
     'computer.cacr.caltech.edu':'.cshrc',
  }
"""

def get_profile(rhost, assume=True):
  '''get the default $PROFILE for a remote host'''
  if _profiles.has_key(rhost):
    return _profiles[rhost]
  if assume:
    return default_profile
  return 


def get_profiles():
  '''get $PROFILE for each registered host'''
  return _profiles


def register_profiles(profiles):
  '''add dict of {'host':$PROFILE} to registered host profiles'''
  #XXX: needs parse checking of input
  _profiles.update(profiles)
  return


def register(rhost, profile=None):
  '''register a host and $PROFILE'''
  if profile == None:
    profile = default_profile
  #XXX: needs parse checking of input
  _profiles[rhost] = profile
  return 


# EOF
