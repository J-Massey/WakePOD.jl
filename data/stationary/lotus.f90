program stationary_foil
  !
    use bodyMod,    only: body
    use fluidMod,   only: fluid
    use mympiMod,   only: init_mympi,mympi_end,mympi_rank
    use gridMod,    only: xg,composite
    use imageMod,   only: display
    use geom_shape
    implicit none
  !
  ! -- Physical parameters
    real,parameter     :: Re = 10250
  !
    real,parameter     :: L=2048, nu=L/Re
    real, parameter    :: finish=70
    integer            :: b(3) = [8,8,1]
  !
  ! -- Hyperparameters
    real, parameter    :: thicc=0.03*L
  !
  ! -- Dimensions
    integer            :: n(3), ndims=2
  !
  ! -- Setup solver
    logical            :: there = .false., root, p(3) = [.FALSE.,.FALSE.,.TRUE.]
    real               :: m(3), z
    type(fluid)        :: flow
    type(body)         :: geom
    !
    ! -- Outputs
      real            :: dt, t, pforce(3),vforce(3),ppower,enstrophy_body,tke_body
    !
    ! -- Initialize
      call init_mympi(ndims,set_blocks=b(1:ndims),set_periodic=p(1:ndims))
      root = mympi_rank()==0
      !
      if(root) print *,'Setting up the grid, body and fluid'
      if(root) print *,'-----------------------------------'
    !
      m = [1.0, 1.0, 0.]
      n = composite(L*m,prnt=root)
      call xg(1)%stretch(n(1), -1.5*L, -.35*L, 2.*L, 5.75*L, h_min=4., h_max=18., prnt=root)
      call xg(2)%stretch(n(2), -1.75*L, -0.35*L, 0.35*L, 1.75*L, prnt=root)
    !
    ! -- Call the geometry and kinematics
      eps=1.
      geom = upper(L,thicc).and.lower(L,thicc).and.wavy_wall(L,thicc)
    !
    ! -- Initialise fluid
      call flow%init(n/b,geom,V=[1.,0.,0.],nu=nu,exit=.true.)
      ! flow%time = 0
    !
      if(root) print *,'Starting time update loop'
      if(root) print *,'-----------------------------------'
    !
      time_loop: do while (flow%time<finish*L.and..not.there)
        t = flow%time
        dt = flow%dt
        call geom%update(t+dt) ! update geom
        call flow%update(geom)
  
        write(9,'(f10.4,f8.4,4e16.8,4e16.8,4e16.8,4e16.8)')&
              t/L,dt,pforce,ppower,vforce,enstrophy_body,tke_body
  
        pforce = 2.*geom%pforce(flow%pressure)/(L*n(3)*xg(3)%h)
        ppower = 2.*geom%ppower(flow%pressure)/(L*n(3)*xg(3)%h)
        vforce = 2.*nu*geom%vforce_s(flow%velocity)/(L*n(3)*xg(3)%h)
        enstrophy_body = flow%velocity%enstrophy(lcorn=L*[-0.25,-0.5,0.],ucorn=L*[2.0,0.5,0.125])
        tke_body = flow%velocity%tke(lcorn=L*[-0.25,-0.5,0.],ucorn=L*[2.0,0.5,0.125])
  
        if((mod(t,L)<dt).and.(root)) print "('Time:',f15.3)",&
        t/L
  
        inquire(file='.kill', exist=there)
        if (there) exit time_loop
        if((t>(finish-17)*L).and.(mod(t,0.005*L)<dt)) call flow%write(geom, write_vtr=.false.)
      end do time_loop
      
      if(root) print *,'Loop complete: writing restart files and exiting'
      if(root) print *,'-----------------------------------'
      ! call flow%write(geom, write_vtr=.true.)
    call mympi_end
  contains
  !
  type(set) function wavy_wall(length, thickness) result(geom)
    real,intent(in) :: length, thickness
    geom = plane([1.,-0.,0.],[length,0.,0.]) & ! end cap
    .and.plane([-1.,0.,0.],[0.,0.,0.]) ! front cap
  end function
  !
  type(set) function upper(length, thickness) result(geom)
    real,intent(in) :: length, thickness
    geom = plane([0.,1.,0.],[0.,0.,0.])&
    .map.init_warp(2,naca_warp,dotnaca_warp,dnaca_warp)
  end function
  !
  type(set) function lower(length, thickness) result(geom)
    real,intent(in) :: length, thickness
    geom = plane([-0.,-1.,0.],[0.,0.,0.])&
    .map.init_warp(2,naca_warp_neg,dotnaca_warp_neg,dnaca_warp_neg)
  end function
  !
  ! -- Create NACA shape by warping based on sharpened NACA0012 profile
  real pure function naca_warp(x)
    real,intent(in) :: x(3)
    real :: xp, a, b, c, d, e, f, g, h, i, j
    a = 0.6128808410319363
    b = -0.48095987091980424
    c = -28.092340603952525
    d = 222.4879939829765
    e = -846.4495017866838
    f = 1883.671432625102
    g = -2567.366504265927
    h = 2111.011565214803
    i = -962.2003374868311
    j = 186.80721148226274
    xp = min(max(x(1)/L,0.),1.)
    naca_warp = L*(a * xp    &    
                  + b * xp**2&
                  + c * xp**3&
                  + d * xp**4&
                  + e * xp**5&
                  + f * xp**6&
                  + g * xp**7&
                  + h * xp**8&
                  + i * xp**9&
                  + j * xp**10)
    
  end function naca_warp
  pure function dnaca_warp(x)
    real,intent(in) :: x(3)
    real            :: dnaca_warp(3)
    real :: xp, a, b, c, d, e, f, g, h, i, j
    a = 0.6128808410319363
    b = -0.48095987091980424
    c = -28.092340603952525
    d = 222.4879939829765
    e = -846.4495017866838
    f = 1883.671432625102
    g = -2567.366504265927
    h = 2111.011565214803
    i = -962.2003374868311
    j = 186.80721148226274
    xp = min(max(x(1)/L,0.),1.)
    dnaca_warp = 0
    dnaca_warp(1) = (a             &    
                      + 2 * b * xp   &
                      + 3 * c * xp**2&
                      + 4 * d * xp**3&
                      + 5 * e * xp**4&
                      + 6 * f * xp**5&
                      + 7 * g * xp**6&
                      + 8 * h * xp**7&
                      + 9 * i * xp**8&
                      + 10 * j * xp**9)
  end function dnaca_warp
  real pure function dotnaca_warp(x)
    real,intent(in) :: x(3)
    dotnaca_warp = 0
  end function dotnaca_warp
  real pure function naca_warp_neg(x)
    real,intent(in) :: x(3)
    real :: xp, a, b, c, d, e, f, g, h, i, j
    a = 0.6128808410319363
    b = -0.48095987091980424
    c = -28.092340603952525
    d = 222.4879939829765
    e = -846.4495017866838
    f = 1883.671432625102
    g = -2567.366504265927
    h = 2111.011565214803
    i = -962.2003374868311
    j = 186.80721148226274
    xp = min(max(x(1)/L,0.),1.)
    naca_warp_neg = -L*(a * xp    &    
                  + b * xp**2&
                  + c * xp**3&
                  + d * xp**4&
                  + e * xp**5&
                  + f * xp**6&
                  + g * xp**7&
                  + h * xp**8&
                  + i * xp**9&
                  + j * xp**10)
                
  end function naca_warp_neg
  pure function dnaca_warp_neg(x)
    real,intent(in) :: x(3)
    real            :: dnaca_warp_neg(3)
    real :: xp, a, b, c, d, e, f, g, h, i, j
    a = 0.6128808410319363
    b = -0.48095987091980424
    c = -28.092340603952525
    d = 222.4879939829765
    e = -846.4495017866838
    f = 1883.671432625102
    g = -2567.366504265927
    h = 2111.011565214803
    i = -962.2003374868311
    j = 186.80721148226274
    xp = min(max(x(1)/L,0.),1.)
    dnaca_warp_neg = 0
    dnaca_warp_neg(1) = -(a            &    
                      + 2 * b * xp   &
                      + 3 * c * xp**2&
                      + 4 * d * xp**3&
                      + 5 * e * xp**4&
                      + 6 * f * xp**5&
                      + 7 * g * xp**6&
                      + 8 * h * xp**7&
                      + 9 * i * xp**8&
                      + 10 * j * xp**9)
  
  end function dnaca_warp_neg
  real pure function dotnaca_warp_neg(x)
    real,intent(in) :: x(3)
    dotnaca_warp_neg = 0
  end function dotnaca_warp_neg
end program stationary_foil
