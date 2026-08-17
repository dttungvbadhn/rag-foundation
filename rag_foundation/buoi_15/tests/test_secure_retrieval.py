import unittest
from src.security import can_access,validate_user_roles

class SecurityBoundaryTests(unittest.TestCase):
 def test_guest_denied_hr(self): self.assertFalse(can_access(["Admin","HR"],["Guest"]))
 def test_hr_allowed_hr(self): self.assertTrue(can_access(["Admin","HR"],["HR"]))
 def test_empty_roles_fail_closed(self):
  with self.assertRaises(ValueError): validate_user_roles([])
 def test_unknown_role_rejected(self):
  with self.assertRaises(ValueError): validate_user_roles(["SuperUser"])
if __name__=="__main__":unittest.main()
