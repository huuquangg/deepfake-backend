package service

import (
	"core-banking/internal/dto"
	"core-banking/internal/models"
	"core-banking/internal/repository"
	"fmt"
	"math/rand"
	"time"
)

type AccountService struct {
	accountRepo *repository.AccountRepository
}

func NewAccountService(accountRepo *repository.AccountRepository) *AccountService {
	return &AccountService{
		accountRepo: accountRepo,
	}
}

// GenerateAccountNumber generates a random 10-digit account number
func (s *AccountService) GenerateAccountNumber() string {
	rand.Seed(time.Now().UnixNano())
	return fmt.Sprintf("%010d", rand.Intn(10000000000))
}

// CreateAccount creates a new account for a user
func (s *AccountService) CreateAccount(userID int) error {
	account := &models.Account{
		UserID:        userID,
		AccountNumber: s.GenerateAccountNumber(),
		Balance:       0.0,
		AccountType:   "SAVINGS",
		Status:        "ACTIVE",
	}
	
	return s.accountRepo.CreateAccount(account)
}

// GetAccountInfo gets account information by user ID
func (s *AccountService) GetAccountInfo(userID int) (*dto.AccountDTO, error) {
	account, err := s.accountRepo.GetAccountByUserID(userID)
	if err != nil {
		return nil, err
	}
	
	return dto.ToAccountDTO(account), nil
}

// GetBalance gets account balance by user ID
func (s *AccountService) GetBalance(userID int) (float64, error) {
	account, err := s.accountRepo.GetAccountByUserID(userID)
	if err != nil {
		return 0, err
	}
	
	return account.Balance, nil
}