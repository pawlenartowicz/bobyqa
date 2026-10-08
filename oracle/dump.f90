module dump_mod
!--------------------------------------------------------------------------------------------------!
! I/O-only state-capture instrumentation (bobyqa state v1 bodies — see oracle/README.md).
! Appends to bobyqa_states.dump in the CWD. This module must never compute with, branch on, or
! modify a dumped value: being write-only is what makes the patch incapable of perturbing the
! oracle's FP results (verified bit-identical by the non-perturbation check in oracle/README.md).
!--------------------------------------------------------------------------------------------------!
use, non_intrinsic :: consts_mod, only : RP, IK
implicit none
private
public :: dump_begin, dump_end
public :: dump_scalar_r, dump_scalar_i, dump_scalar_l
public :: dump_vector, dump_matrix, dump_imatrix

! N.B.: NEWUNIT= returns a NEGATIVE unit number (F2008), so the open-once guard must be an
! explicit flag — `funit < 0` would re-open on every call.
integer, save :: funit = -1
logical, save :: file_open = .false.
character(len=*), parameter :: FNAME = 'bobyqa_states.dump'
! 17 significant digits round-trips IEEE-754 f64 exactly.
character(len=*), parameter :: RFMT = 'ES25.16E3'

contains

subroutine ensure_open()
if (.not. file_open) then
    open (newunit=funit, file=FNAME, action='write', position='append', status='unknown')
    file_open = .true.
end if
end subroutine ensure_open

subroutine dump_begin(routine, marker)  ! marker is 'entry' or 'exit'
character(len=*), intent(in) :: routine, marker
call ensure_open()
write (funit, '(A, 1X, A, 1X, A)') 'state', routine, marker
end subroutine dump_begin

subroutine dump_end()
write (funit, '(A)') 'end'
flush (funit)
end subroutine dump_end

subroutine dump_scalar_r(name, v)
character(len=*), intent(in) :: name
real(RP), intent(in) :: v
write (funit, '(A, 1X, A, 1X, '//RFMT//')') 'scalar', name, v
end subroutine dump_scalar_r

subroutine dump_scalar_i(name, v)
character(len=*), intent(in) :: name
integer(IK), intent(in) :: v
write (funit, '(A, 1X, A, 1X, I0)') 'scalar', name, v
end subroutine dump_scalar_i

subroutine dump_scalar_l(name, v)  ! logical dumped as 0/1
character(len=*), intent(in) :: name
logical, intent(in) :: v
write (funit, '(A, 1X, A, 1X, I0)') 'scalar', name, merge(1, 0, v)
end subroutine dump_scalar_l

subroutine dump_vector(name, v)
character(len=*), intent(in) :: name
real(RP), intent(in) :: v(:)
write (funit, '(A, 1X, A, *(1X, '//RFMT//'))') 'vector', name, v
end subroutine dump_vector

subroutine dump_matrix(name, a)  ! flattened column-major (Fortran natural order)
character(len=*), intent(in) :: name
real(RP), intent(in) :: a(:, :)
write (funit, '(A, 1X, A, 1X, I0, 1X, I0, *(1X, '//RFMT//'))') 'matrix', name, size(a, 1), size(a, 2), a
end subroutine dump_matrix

subroutine dump_imatrix(name, a)
character(len=*), intent(in) :: name
integer(IK), intent(in) :: a(:, :)
write (funit, '(A, 1X, A, 1X, I0, 1X, I0, *(1X, I0))') 'imatrix', name, size(a, 1), size(a, 2), a
end subroutine dump_imatrix

end module dump_mod
